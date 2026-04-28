import warnings
# Suppress WandB/Pydantic compatibility warnings
warnings.filterwarnings("ignore", message=".*'repr' attribute.*")
warnings.filterwarnings("ignore", message=".*'frozen' attribute.*")

import os
from typing import Union, List, Tuple
import torch
import timm
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
from forgeryscope.model_zoo import MODEL_SPECS, get_model_path


class AttentionPooling(torch.nn.Module):
    """Attention-based pooling - learns which spatial regions are important."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.attention = torch.nn.Sequential(
            torch.nn.Conv2d(in_dim, in_dim // 4, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_dim // 4, 1, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.attention(x)
        attn = attn / (attn.sum(dim=(-2, -1), keepdim=True) + 1e-8)
        pooled = (x * attn).sum(dim=(-2, -1))
        return pooled


class Embedder:
    def __init__(
        self, 
        checkpoint_path: str, 
        device: str = 'cuda', 
        width: int = None, 
        height: int = None, 
        transform_type: str = None,
        model_base_url: str = None,
        cache_dir: str = None,
        verbose: bool = False,
    ):
        self.device = device
        self.verbose = verbose
        checkpoint_spec = MODEL_SPECS.get(checkpoint_path, {})
        width = width or int(checkpoint_spec.get("width", 224))
        height = height or int(checkpoint_spec.get("height", 224))
        transform_type = transform_type or str(checkpoint_spec.get("transform_type", "resize"))
        checkpoint_path = get_model_path(checkpoint_path, cache_dir=cache_dir, base_url=model_base_url)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        hparams = checkpoint['hyper_parameters']
        state_dict = checkpoint['state_dict']
        backbone_state = {k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone.')}
        custom_pool_state = {k.replace('custom_pool.', ''): v for k, v in state_dict.items() if k.startswith('custom_pool.')}

        pooling_type = hparams.get('pooling_type', 'avg')
        if self.verbose:
            print(f"Using pooling type: {pooling_type}")
        
        if pooling_type in ["none", "attention"]:
            global_pool_setting = ""
        else:
            global_pool_setting = "avg"
        
        self.backbone = timm.create_model(
            hparams['model_name'], 
            pretrained=False, 
            num_classes=0, 
            global_pool=global_pool_setting
        )
        
        if hasattr(self.backbone, 'num_features'):
            feat_dim = self.backbone.num_features
        else:
            # Try to infer from dummy input
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, height, width)
                if pooling_type in ["none", "gem", "attention", "multiscale"]:
                    features = self.backbone.forward_features(dummy_input)
                    if isinstance(features, (list, tuple)):
                        features = features[-1]
                    feat_dim = features.shape[1]
                else:
                    features = self.backbone(dummy_input)
                    feat_dim = features.shape[1]
        
        self.backbone.load_state_dict(backbone_state)
        self.backbone.to(device).eval()
        
        self.pooling_type = pooling_type
        self.custom_pool = None
        if pooling_type == "attention":
            self.custom_pool = AttentionPooling(feat_dim)
            if len(custom_pool_state):
                self.custom_pool.load_state_dict(custom_pool_state)
            self.custom_pool.to(device).eval()

        self.projection = torch.nn.Identity()

        model_config = timm.data.resolve_model_data_config(hparams['model_name'])
        norm_mean = model_config.get("mean")
        norm_std = model_config.get("std")
        
        if self.verbose:
            print(f"Model : {hparams['model_name']}. Using normalization mean: {norm_mean}, std: {norm_std}")
            print('using transform_type:', transform_type)
        if transform_type == 'torchvision_resize':
            self.transform = TF.Compose([
                TF.Resize((height, width)),
                TF.ToTensor(),
                TF.Normalize(mean=norm_mean, std=norm_std),
            ])
        elif transform_type == 'resize':
            self.transform = A.Compose([
                A.Resize(height, width),
                A.Normalize(mean=norm_mean, std=norm_std),
                ToTensorV2()
            ])
        elif transform_type == 'longest_max_size':
            self.transform = A.Compose([
                A.LongestMaxSize(max_size=width),
                A.PadIfNeeded(
                    min_height=height,
                    min_width=width,
                    border_mode=0,
                ),
                A.Resize(height, width),
                A.Normalize(mean=norm_mean, std=norm_std),
                ToTensorV2()
            ])
        else:
            raise ValueError(f"Unknown transform_type: {transform_type}")
        self.transform_type = transform_type
    
    def transform_image(self, img: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Transform image for model input.
        
        Args:
            img: Image as file path (str), PIL Image, or numpy array (HWC uint8)
            
        Returns:
            torch.Tensor: Preprocessed image tensor [1, C, H, W] on device
        """
        # Check if using albumentations transform (resize or longest_max_size)
        is_albu_transform = self.transform_type in ['resize', 'longest_max_size']
        
        if is_albu_transform:
            if isinstance(img, Image.Image):
                img = np.array(img)
            img_tensor = self.transform(image=img)['image'].unsqueeze(0).to(self.device)
        else:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img.astype(np.uint8))
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        return img_tensor
    
    def _get_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial feature maps from backbone.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Spatial feature maps [B, C, H, W]
        """
        if hasattr(self.backbone, 'forward_features'):
            features = self.backbone.forward_features(x)
        else:
            raise NotImplementedError(
                f"Model {type(self.backbone).__name__} does not support forward_features. "
                f"Consider using a different model or pooling_type='avg'."
            )
        
        # Handle different return types
        if isinstance(features, (list, tuple)):
            features = features[-1]  # Take the last feature map (highest resolution)
        
        return features  # [B, C, H, W]
    
    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that handles both standard and custom pooling.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Pooled features [B, D]
        """
        if self.pooling_type == "none":
            # Get spatial feature maps
            features = self._get_spatial_features(x)
            # Flatten spatial dimensions and average
            B, C, H, W = features.shape
            features = features.view(B, C, -1).mean(dim=-1)  # [B, C]
        elif self.custom_pool is not None:
            # Get spatial feature maps and apply custom pooling
            features = self._get_spatial_features(x)
            features = self.custom_pool(features)  # [B, C]
        else:
            # Use backbone's default pooling (avg)
            features = self.backbone(x)  # [B, C]
        
        # Ensure features are 2D
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        
        return features
    
    @torch.no_grad()
    def get_embedding(self, img: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Get embedding for a single image.
        
        Args:
            img: Image as file path (str), PIL Image, or numpy array (HWC uint8)
            
        Returns:
            torch.Tensor: Normalized embedding [1, D]
        """
        img_tensor = self.transform_image(img)
        features = self._forward_features(img_tensor)
        embedding = self.projection(features)
        return F.normalize(embedding, p=2, dim=1)


    @torch.no_grad()
    def get_embedding_batch(self, imgs_np: List[Union[str, Image.Image, np.ndarray]]) -> torch.Tensor:
        """
        Get embeddings for a batch of images.
        
        Args:
            imgs_np: List of images as file paths (str), PIL Images, or numpy arrays (HWC uint8)
            
        Returns:
            torch.Tensor: Normalized embeddings [B, D]
        """
        if not isinstance(imgs_np, (list, tuple)) or len(imgs_np) == 0:
            raise ValueError("imgs_np must be a non-empty list or tuple")

        batch = torch.cat([self.transform_image(img) for img in imgs_np], dim=0)
        feats = self._forward_features(batch)
        emb = self.projection(feats)
        return F.normalize(emb, dim=1)

    @torch.no_grad()
    def get_embedding_batch_torch(self, imgs_tensor: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for a batch of preprocessed images.
        
        Args:
            imgs_tensor: Preprocessed image tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Normalized embeddings [B, D]
        """
        imgs_tensor = imgs_tensor.to(self.device)
        feats = self._forward_features(imgs_tensor)
        emb = self.projection(feats)
        return F.normalize(emb, dim=1)
    
    @torch.no_grad()
    def compare(
        self, 
        pil_image1: Union[str, Image.Image, np.ndarray], 
        pil_image2: Union[str, Image.Image, np.ndarray]
    ) -> float:
        """
        Compare two images. Returns similarity score in [0, 1].
        
        Args:
            pil_image1: First image as file path (str), PIL Image, or numpy array
            pil_image2: Second image as file path (str), PIL Image, or numpy array
            
        Returns:
            float: Cosine similarity score in [0, 1]
        """
        emb1 = self.get_embedding(pil_image1)
        emb2 = self.get_embedding(pil_image2)
        
        similarity = F.cosine_similarity(emb1, emb2).item()
        return similarity
    
    @staticmethod
    def find_similar_pairs(
        embeddings: Union[torch.Tensor, np.ndarray], 
        threshold: float = 0.9
    ) -> List[Tuple[int, int, float]]:
        """
        Find similar pairs from embeddings based on cosine similarity.
        
        Args:
            embeddings: Tensor or numpy array of shape [N, D]
            threshold: Similarity threshold (default: 0.9)
            
        Returns:
            List of tuples (i, j, similarity) for pairs above threshold, sorted by similarity
        """
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
        
        similarity_matrix = embeddings @ embeddings.T
        
        pairs = []
        n = similarity_matrix.shape[0]
        
        for i in range(n):
            for j in range(i + 1, n):  # Only upper triangle, no self-similarity
                sim = similarity_matrix[i, j].item()
                if sim >= threshold:
                    pairs.append((i, j, sim))
        
        # Sort by similarity (highest first)
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        return pairs



class TorchImageDataset(Dataset):
    def __init__(
        self,
        pathes,
        transform=None,
        img_dir=None
    ):
        self.pathes = pathes
        self.transform = transform 
        self.is_albu = isinstance(transform, A.Compose)
        self.img_dir = img_dir
        print(f"TorchImageDataset: num images = {len(self.pathes)}, img_dir = {self.img_dir}, is_albu = {self.is_albu}")

    def __len__(self):
        return len(self.pathes)

    def _load_image(self, path: str) -> Image.Image:
        if self.img_dir:
            path = os.path.join(self.img_dir, path)
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        path = self.pathes[idx]
        img = self._load_image(path)
        if not self.is_albu:
            img_torch = self.transform(img)
        else:
            img_torch = self.transform(image=np.array(img))['image']
        return img_torch, path
