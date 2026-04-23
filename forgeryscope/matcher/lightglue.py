from typing import Union, List, Tuple, Dict, Optional, Any
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
from lightglue import LightGlue, SIFT, ALIKED, match_pair
from lightglue.utils import load_image, numpy_image_to_torch
from shapely.geometry import Polygon, box
from shapely.affinity import translate
import pandas as pd
from forgeryscope.matcher.geometry import (
    clip_polygon_to_image,
    polygon_to_bbox_mask,
    polygon_to_mask,
    polygon_to_bbox)

torch.set_grad_enabled(False)


class LightGlueOverlap:
    """
    LightGlue-based image overlap computation and matching.
    
    This class uses LightGlue feature matching to compute overlap regions
    between pairs of images, with support for various feature extractors
    (ALIKED, SIFT) and robust transformation estimation (RANSAC, MAGSAC).
    
    Example:
        matcher = LightGlueOverlap(
            max_keypoints=2048,
            matcher_features="aliked",
            device="cuda"
        )
        result = matcher.compute_overlap(img1, img2, test_transforms=True)
    """
    
    def __init__(
        self, 
        max_keypoints: int = 2048, 
        matcher_features: str = "superpoint", 
        device: str = "cuda", 
        depth_confidence: float = 0.95, 
        width_confidence: float = 0.99, 
        filter_threshold: float = 0.1, 
        model_name_aliked: str = "aliked-n16", 
        estimator_method: str = "RANSAC", 
        reprojThreshold: float = 5.0, 
        estimator_confidence: float = 0.9999, 
        estimator_maxIters: int = 5000, 
        estimator_refineIters: int = 10,
        extractor_resize: Optional[int] = None
    ):
        self.device = torch.device(device)


        if matcher_features.lower() == "aliked":
            self.extractor = ALIKED(model_name=model_name_aliked, max_num_keypoints=max_keypoints).eval().to(self.device)
            print(f"Using ALIKED {model_name_aliked} with max {max_keypoints} keypoints.")
        elif matcher_features.lower() == "sift":
            self.extractor = SIFT()
        else:
            raise ValueError(f"Unknown feature extractor: {matcher_features}. Supported: 'aliked', 'sift'")
        self.matcher = LightGlue(features=matcher_features, filter_threshold=filter_threshold, depth_confidence=depth_confidence, width_confidence=width_confidence).eval().to(self.device)
        self.estimator_method = estimator_method
        self.reprojThreshold = reprojThreshold
        self.estimator_confidence = estimator_confidence
        self.estimator_maxIters = estimator_maxIters
        self.estimator_refineIters = estimator_refineIters
        print(f"Estimator method: {self.estimator_method} with reprojThreshold: {self.reprojThreshold}, confidence: {self.estimator_confidence}, maxIters: {self.estimator_maxIters}, refineIters: {self.estimator_refineIters}.")
        if self.estimator_method not in {"RANSAC", "MAGSAC"}:
            raise NotImplementedError("Currently only RANSAC and MAGSAC are supported.")
        print(f"LightGlue matcher initialized with features: {matcher_features}, filter_threshold: {filter_threshold}, depth_confidence: {depth_confidence}, width_confidence: {width_confidence}.")
        print(f"LightGlueOverlap loaded. Device: {self.device}, Max keypoints: {max_keypoints}")
        self.extractor_resize = extractor_resize
        if extractor_resize is not None:
            print(f"Extractor resize enabled: {self.extractor_resize} !!")

    
    def _match_features(
        self, 
        image0: torch.Tensor, 
        image1: torch.Tensor
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
        """
        Run LightGlue matching between two images.
        
        Args:
            image0: First image tensor [C, H, W]
            image1: Second image tensor [C, H, W]
        
        Returns:
            Tuple of (pts0, pts1, match_scores, error_msg):
            - pts0: Matched keypoints in image0 [N, 2] or None if failed
            - pts1: Matched keypoints in image1 [N, 2] or None if failed
            - match_scores: Match confidence scores [N] or None if failed
            - error_msg: Error message string or None if successful
        """
        try:
            feats0, feats1, matches01 = match_pair(self.extractor, self.matcher, image0, image1, resize=self.extractor_resize)

            kpts0 = feats0["keypoints"]
            kpts1 = feats1["keypoints"]
            matches = matches01["matches"]
            scores = matches01["scores"]
            torch.cuda.synchronize()

            if matches is None or len(matches) < 4:
                return None, None, None, "Not enough matches: {}".format(len(matches) if matches is not None else 0)
        except Exception as e:
            return None, None, None, f"Matching failed with error: {e}"
        
        pts0 = kpts0[matches[:, 0]].cpu().numpy().astype(np.float32)
        pts1 = kpts1[matches[:, 1]].cpu().numpy().astype(np.float32)
        match_scores = scores.cpu().numpy() 

        return pts0, pts1, match_scores, None

    def _prepare_image(
        self, 
        img: Union[str, Path, Image.Image, np.ndarray]
    ) -> torch.Tensor:
        """
        Prepare image for processing by converting to torch tensor.
        
        Args:
            img: Image as file path (str/Path), PIL Image, or numpy array (HWC uint8)
        
        Returns:
            torch.Tensor: Image tensor [C, H, W] in [0, 1] range
        
        Raises:
            ValueError: If image type is not supported
        """
        if isinstance(img, (str, Path)):
            return load_image(img)
        elif isinstance(img, Image.Image):
            img_np = np.array(img)
            return numpy_image_to_torch(img_np)
        elif isinstance(img, np.ndarray):
            return numpy_image_to_torch(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}. Supported: str, Path, PIL.Image, np.ndarray")
    
    def _apply_image_transform(
        self, 
        image_tensor: torch.Tensor, 
        transform_type: str
    ) -> torch.Tensor:
        """
        Apply transformation to image tensor.
        
        Args:
            image_tensor: torch.Tensor of shape (C, H, W)
            transform_type: str, one of 'original', 'fliplr', 'flipud', 'rot180'
        
        Returns:
            torch.Tensor: Transformed image tensor of same shape
        
        Raises:
            ValueError: If transform_type is not supported
        """
        if transform_type == 'original':
            return image_tensor
        elif transform_type == 'fliplr':
            # Flip left-right: flip along width dimension
            return torch.flip(image_tensor, dims=[2])
        elif transform_type == 'flipud':
            # Flip up-down: flip along height dimension
            return torch.flip(image_tensor, dims=[1])
        elif transform_type == 'rot180':
            # Rotate 180 degrees: flip both dimensions
            return torch.flip(image_tensor, dims=[1, 2])
        else:
            raise ValueError(f"Unknown transform_type: {transform_type}")
    
    def _get_transform_matrix(
        self, 
        transform_type: str, 
        height: int, 
        width: int
    ) -> np.ndarray:
        """
        Get the transformation matrix that maps original coordinates to transformed coordinates.
        
        Args:
            transform_type: str, one of 'original', 'fliplr', 'flipud', 'rot180'
            height: int, image height
            width: int, image width
        
        Returns:
            np.ndarray: Transformation matrix of shape (3, 3), dtype float32
        
        Raises:
            ValueError: If transform_type is not supported
        """
        if transform_type == 'original':
            return np.eye(3, dtype=np.float32)
        elif transform_type == 'fliplr':
            # (x, y) -> (W - x, y)
            T = np.array([[-1, 0, width],
                          [0, 1, 0],
                          [0, 0, 1]], dtype=np.float32)
        elif transform_type == 'flipud':
            # (x, y) -> (x, H - y)
            T = np.array([[1, 0, 0],
                          [0, -1, height],
                          [0, 0, 1]], dtype=np.float32)
        elif transform_type == 'rot180':
            # (x, y) -> (W - x, H - y)
            T = np.array([[-1, 0, width],
                          [0, -1, height],
                          [0, 0, 1]], dtype=np.float32)
        else:
            raise ValueError(f"Unknown transform_type: {transform_type}")
        return T
    
    def _compute_overlap_single(
        self, 
        image0: torch.Tensor, 
        image1_transformed: torch.Tensor, 
        transform_type: str
    ) -> Dict[str, Any]:
        """
        Compute overlap for a single transformed version of image1.
        
        Args:
            image0: torch.Tensor, original image0 [C, H, W]
            image1_transformed: torch.Tensor, transformed image1 [C, H, W]
            transform_type: str, the transform applied to image1
        
        Returns:
            dict: Overlap results containing:
                - 'H': Homography matrix mapping image1 to image0 [3, 3]
                - 'H_transformed': Homography for transformed image1 [3, 3]
                - 'transform_type': Applied transform type
                - 'inliers': Number of inlier matches
                - 'total_matches': Total number of matches
                - 'match_scores': Array of match scores
                - 'mean_match_score': Mean match score
                - 'median_match_score': Median match score
                - 'min_match_score': Minimum match score
                - 'inlier_mean_score': Mean score of inliers
                - 'overlap_poly_img0': Shapely Polygon in image0 coordinates
                - 'overlap_poly_img1': Shapely Polygon in image1 coordinates
                - 'bbox_img0': Bounding box (minx, miny, maxx, maxy) in image0
                - 'bbox_img1': Bounding box (minx, miny, maxx, maxy) in image1
                - 'overlap_area_img0': Overlap area in image0 pixels
                - 'overlap_area_img1': Overlap area in image1 pixels
                - 'percent_img0': Overlap percentage of image0
                - 'percent_img1': Overlap percentage of image1
                Or error dict with 'error' key if computation failed
        """
        # Match features
        pts0, pts1_transformed, match_scores, err = self._match_features(image0, image1_transformed)
        if err:
            return {"error": err}
        
        # Estimate transformation
        if self.estimator_method == "RANSAC":
            M_transformed, mask = cv2.estimateAffinePartial2D(
                pts1_transformed, pts0,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.reprojThreshold,
                confidence=self.estimator_confidence,
                maxIters=self.estimator_maxIters,
                refineIters=self.estimator_refineIters)
        else:
            M_transformed, mask = cv2.estimateAffine2D(
                pts1_transformed, pts0,
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=self.reprojThreshold,
                confidence=self.estimator_confidence,
                maxIters=self.estimator_maxIters,
                refineIters=self.estimator_refineIters)
        
        if M_transformed is None:
            return {"error": "Affine transformation computation failed"}
        
        H_transformed = np.vstack([M_transformed, [0, 0, 1]])
        
        # Get image dimensions
        h0, w0 = image0.shape[-2:]
        h1, w1 = image1_transformed.shape[-2:]
        
        # Build polygons in transformed space
        corners0 = np.float32([[0,0],[w0,0],[w0,h0],[0,h0]]).reshape(-1,1,2)
        corners1_transformed = np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)
        
        proj_corners1_transformed = cv2.perspectiveTransform(corners1_transformed, H_transformed)
        
        poly0 = Polygon(corners0.reshape(-1,2))
        poly1_transformed = Polygon(proj_corners1_transformed.reshape(-1,2))
        if not poly1_transformed.is_valid:
            poly1_transformed = poly1_transformed.buffer(0)
        
        # Intersection in image0 coordinates
        overlap0 = poly0.intersection(poly1_transformed)
        if overlap0.is_empty:
            return {"error": "No overlap found"}
        
        # Choose largest polygon if multipolygon
        if overlap0.geom_type == "MultiPolygon":
            overlap0 = max(overlap0.geoms, key=lambda p: p.area)
        
        # Transform overlap polygon back to original image1 coordinates
        # T maps original_img1 -> transformed_img1
        # H_transformed maps transformed_img1 -> img0
        # So H_original = H_transformed @ T maps original_img1 -> img0
        T = self._get_transform_matrix(transform_type, h1, w1)
        H_original = H_transformed @ T
        
        # Transform overlap from img0 coordinates back to original img1 coordinates
        H_original_inv = np.linalg.inv(H_original)
        coords0 = np.array(overlap0.exterior.coords[:-1], dtype=np.float32)
        coords1 = cv2.perspectiveTransform(coords0.reshape(-1,1,2), H_original_inv).reshape(-1,2)
        
        overlap1 = Polygon(coords1)
        if not overlap1.is_valid:
            overlap1 = overlap1.buffer(0)
        
        minx0, miny0, maxx0, maxy0 = overlap0.bounds
        minx1, miny1, maxx1, maxy1 = overlap1.bounds
        
        mean_score = float(match_scores.mean())
        
        return {
            "H": H_original,
            "H_transformed": H_transformed,
            "transform_type": transform_type,
            "inliers": int(mask.sum()),
            "total_matches": len(mask),
            "match_scores": match_scores,
            "mean_match_score": mean_score,
            "median_match_score": float(np.median(match_scores)),
            "min_match_score": float(match_scores.min()),
            "inlier_mean_score": float(match_scores[mask.ravel().astype(bool)].mean()),
            "overlap_poly_img0": overlap0,
            "overlap_poly_img1": overlap1,
            "bbox_img0": (minx0, miny0, maxx0, maxy0),
            "bbox_img1": (minx1, miny1, maxx1, maxy1),
            "overlap_area_img0": overlap0.area,
            "overlap_area_img1": overlap1.area,
            "percent_img0": 100 * overlap0.area / (w0 * h0),
            "percent_img1": 100 * overlap1.area / (w1 * h1),
        }

    def compute_overlap(
        self, 
        img_path_0: Union[str, Path, Image.Image, np.ndarray], 
        img_path_1: Union[str, Path, Image.Image, np.ndarray], 
        test_transforms: bool = False
    ) -> Dict[str, Any]:
        """
        Main method to compute overlap bboxes & stats between two images.
        
        Args:
            img_path_0: First image as file path (str/Path), PIL Image, or numpy array
            img_path_1: Second image as file path (str/Path), PIL Image, or numpy array
            test_transforms: If True, tests 4 variations of img1 (original, fliplr, flipud, rot180) 
                           and selects the best match. If False (default), uses original images only.
        
        Returns:
            dict: Overlap results (see _compute_overlap_single for structure) or error dict with 'error' key
        
        Example:
            >>> matcher = LightGlueOverlap()
            >>> result = matcher.compute_overlap("img1.jpg", "img2.jpg")
            >>> if "error" not in result:
            ...     print(f"Overlap: {result['percent_img0']:.2f}% of image0")
        """
        # Load images
        image0 = self._prepare_image(img_path_0).to(self.device)
        image1 = self._prepare_image(img_path_1).to(self.device)
        
        if not test_transforms:
            # Original behavior: just match original images
            # Use _compute_overlap_single with 'original' transform (which is identity)
            result = self._compute_overlap_single(image0, image1, 'original')
            # Remove transform-specific fields for backward compatibility
            if "error" not in result:
                result.pop("H_transformed", None)
                result.pop("transform_type", None)
            return result
        
        # Enhanced behavior: test all 4 variations of image1
        transform_types = ['original', 'fliplr', 'flipud', 'rot180']
        best_result = None
        best_score = -np.inf
        
        for transform_type in transform_types:
            # Apply transformation to image1
            image1_transformed = self._apply_image_transform(image1, transform_type)
            
            # Compute overlap for this variation
            result = self._compute_overlap_single(image0, image1_transformed, transform_type)
            
            if "error" not in result:
                score = result['mean_match_score']
                if score > best_score:
                    best_score = score
                    best_result = result
        
        # If no variation succeeded, return error
        if best_result is None:
            return {"error": "All transform variations failed to find matches"}
        
        # Return the best result (already mapped back to original coordinates)
        return best_result

    def visualize(
        self, 
        img0: Union[str, Path, Image.Image, np.ndarray], 
        img1: Union[str, Path, Image.Image, np.ndarray], 
        overlap_result: Dict[str, Any], 
        to_bbox: bool = False
    ) -> None:
        """
        Visualize overlap regions on two images side by side.
        
        Args:
            img0: First image as file path (str/Path), PIL Image, or numpy array
            img1: Second image as file path (str/Path), PIL Image, or numpy array
            overlap_result: Result dictionary from compute_overlap()
            to_bbox: If True, convert polygons to bounding boxes before visualization
        
        Raises:
            KeyError: If overlap_result is missing required keys
        """
        # Validate required keys
        required_keys = ["overlap_poly_img0", "overlap_poly_img1"]
        missing_keys = [key for key in required_keys if key not in overlap_result]
        if missing_keys:
            raise KeyError(f"overlap_result missing required keys: {missing_keys}")
        
        image0 = self._prepare_image(img0).permute(1,2,0).cpu().numpy()
        image1 = self._prepare_image(img1).permute(1,2,0).cpu().numpy()

        poly0 = overlap_result["overlap_poly_img0"]
        poly1 = overlap_result["overlap_poly_img1"]
        if to_bbox:
            bbox0 = polygon_to_bbox(poly0)
            bbox1 = polygon_to_bbox(poly1)
            poly0 = Polygon([(bbox0[0], bbox0[1]), (bbox0[2], bbox0[1]), (bbox0[2], bbox0[3]), (bbox0[0], bbox0[3])])
            poly1 = Polygon([(bbox1[0], bbox1[1]), (bbox1[2], bbox1[1]), (bbox1[2], bbox1[3]), (bbox1[0], bbox1[3])])

        poly0 = clip_polygon_to_image(poly0, image0.shape[0], image0.shape[1])
        poly1 = clip_polygon_to_image(poly1, image1.shape[0], image1.shape[1])
        fig, (ax0, ax1) = plt.subplots(1,2,figsize=(14,7))

        # Image0
        ax0.imshow(image0)
        ax0.plot(*poly0.exterior.xy, color="red")
        ax0.set_title("Image0 Overlap")
        ax0.axis("off")

        # Image1
        ax1.imshow(image1)
        ax1.plot(*poly1.exterior.xy, color="red")
        ax1.set_title("Image1 Overlap")
        ax1.axis("off")

        plt.show()

    def visualize_matches(
        self, 
        img0: Union[str, Path, Image.Image, np.ndarray], 
        img1: Union[str, Path, Image.Image, np.ndarray], 
        max_matches: int = 10000, 
        alpha: float = 0.75, 
        linewidth: float = 0.5
    ) -> None:
        """
        Visualize matched keypoints between two images.
        
        Args:
            img0: First image as file path (str/Path), PIL Image, or numpy array
            img1: Second image as file path (str/Path), PIL Image, or numpy array
            max_matches: Maximum number of matches to display (to avoid clutter)
            alpha: Transparency of match lines (0-1)
            linewidth: Width of match lines
        """        
        # Prepare images
        image0 = self._prepare_image(img0).to(self.device)
        image1 = self._prepare_image(img1).to(self.device)
        
        # Get features and matches
        feats0, feats1, matches01 = match_pair(self.extractor, self.matcher, image0, image1, resize=self.extractor_resize)
        
        kpts0 = feats0["keypoints"].cpu().numpy()
        kpts1 = feats1["keypoints"].cpu().numpy()
        matches = matches01["matches"].cpu().numpy()
        scores = matches01["scores"].cpu().numpy()
        
        # Convert images to numpy for display
        img0_np = image0.permute(1, 2, 0).cpu().numpy()
        img1_np = image1.permute(1, 2, 0).cpu().numpy()
        
        # Limit number of matches for clarity
        if len(matches) > max_matches:
            # Sort by score and take top matches
            top_indices = np.argsort(scores)[-max_matches:]
            matches = matches[top_indices]
            scores = scores[top_indices]
        
        # Create side-by-side visualization
        h0, w0 = img0_np.shape[:2]
        h1, w1 = img1_np.shape[:2]
        h = max(h0, h1)
        
        # Create canvas
        canvas = np.zeros((h, w0 + w1, 3), dtype=np.float32)
        canvas[:h0, :w0] = img0_np
        canvas[:h1, w0:] = img1_np
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        ax.imshow(canvas)
        
        # Draw matches
        for i, (m0, m1) in enumerate(matches):
            pt0 = kpts0[m0]
            pt1 = kpts1[m1] + [w0, 0]  # Offset for second image
            
            # Color by confidence
            color = plt.cm.viridis(scores[i])
            
            # Draw line
            ax.plot([pt0[0], pt1[0]], [pt0[1], pt1[1]], 
                    'r-', linewidth=linewidth, alpha=alpha, color=color)
            
            # Draw keypoints
            ax.plot(pt0[0], pt0[1], 'o', markersize=3, color=color)
            ax.plot(pt1[0], pt1[1], 'o', markersize=3, color=color)
        
        ax.set_title(f'Matched Keypoints: {len(matches)} matches\n'
                    f'Mean confidence: {scores.mean():.3f}')
        ax.axis('off')
        plt.tight_layout()
        plt.show()
        
        print(f"Total matches: {len(matches01['matches'])}")
        print(f"Displayed: {len(matches)}")
        print(f"Mean match score: {scores.mean():.3f}")


def create_duplicate_masks(
    img: np.ndarray,
    panels: List[Tuple[Any, ...]],
    crops_list: List[np.ndarray],
    clf_predicts: pd.DataFrame,
    matcher_micro: LightGlueOverlap,
    matcher_blot: LightGlueOverlap,
    to_bbox_micro: bool = False,
    to_bbox_blot: bool = False,
    fallback_for_wblot: bool = False,
    test_transforms_micro: bool = False,
    test_transforms_blot: bool = False
) -> List[Dict[str, Any]]:
    """
    Create duplicate masks for panel pairs based on classifier predictions.
    
    Args:
        img: Source image as numpy array (H, W, C)
        panels: List of panel tuples (label, conf, x0, y0, x1, y1) or with idx
        crops_list: List of cropped panel images as numpy arrays
        clf_predicts: DataFrame with columns ['idx1', 'idx2', 'label'] for duplicate pairs
        matcher_micro: LightGlueOverlap instance for microscopy/body imaging panels
        matcher_blot: LightGlueOverlap instance for blot panels
        to_bbox_micro: If True, convert overlap polygons to bboxes for microscopy
        to_bbox_blot: If True, convert overlap polygons to bboxes for blots
        fallback_for_wblot: If True, use full panel bbox as fallback when matching fails for blots
        test_transforms_micro: If True, test image transforms for microscopy matching
        test_transforms_blot: If True, test image transforms for blot matching
    
    Returns:
        List of dictionaries, each containing:
            - 'panel_id0', 'panel_id1': Panel indices
            - 'panel_label': Panel type ('Blots', 'Microscopy', 'Body Imaging')
            - 'poly_coords0', 'poly_coords1': Overlap polygon coordinates
            - 'bbox_crop0', 'bbox_crop1': Bounding boxes in crop coordinates
            - 'mask0', 'mask1': Binary masks for overlap regions
            - 'match_result': Match result dictionary from compute_overlap()
    
    Raises:
        ValueError: If panel_label is not recognized
        KeyError: If clf_predicts DataFrame is missing required columns
    """
    # Input validation
    if not isinstance(img, np.ndarray) or len(img.shape) < 2:
        raise ValueError(f"img must be a numpy array with at least 2 dimensions, got {type(img)}")
    if not isinstance(panels, (list, tuple)) or len(panels) == 0:
        raise ValueError(f"panels must be a non-empty list or tuple, got {type(panels)}")
    if not isinstance(crops_list, (list, tuple)) or len(crops_list) != len(panels):
        raise ValueError(f"crops_list must have same length as panels ({len(panels)}), got {len(crops_list)}")
    required_cols = ['idx1', 'idx2', 'label']
    if not all(col in clf_predicts.columns for col in required_cols):
        missing = [col for col in required_cols if col not in clf_predicts.columns]
        raise ValueError(f"clf_predicts missing required columns: {missing}")
    
    height, width = img.shape[:2]
    match_results = []
    for _, row in clf_predicts.iterrows():
        id0 = int(row['idx1'])
        id1 = int(row['idx2'])
        panel_label = row['label']
        
        # Validate panel indices
        if id0 < 0 or id0 >= len(panels):
            raise IndexError(f"Panel index id0={id0} out of range [0, {len(panels)})")
        if id1 < 0 or id1 >= len(panels):
            raise IndexError(f"Panel index id1={id1} out of range [0, {len(panels)})")
        if len(panels[id0]) < 4:
            raise ValueError(f"Panel {id0} tuple too short, expected at least 4 elements (x0, y0, x1, y1)")
        if len(panels[id1]) < 4:
            raise ValueError(f"Panel {id1} tuple too short, expected at least 4 elements (x0, y0, x1, y1)")

        bbox0 = panels[id0][-4:]  # x1,y1,x2,y2 of panel id0
        bbox1 = panels[id1][-4:]

        if panel_label == 'Blots':
            # print('using blot matcher')
            match_result = matcher_blot.compute_overlap(crops_list[id0], crops_list[id1], test_transforms=test_transforms_blot)
        else:
            # print('using micro matcher')
            match_result = matcher_micro.compute_overlap(crops_list[id0], crops_list[id1], test_transforms=test_transforms_micro)

        if "error" in match_result:
            if fallback_for_wblot and (panel_label == 'Blots'):
                # print('Use full bboxes as fallback')
                poly0 = box(bbox0[0], bbox0[1], bbox0[2], bbox0[3])
                poly1 = box(bbox1[0], bbox1[1], bbox1[2], bbox1[3])
                
                bbox_crop0 = (0, 0, bbox0[2] - bbox0[0], bbox0[3] - bbox0[1])
                bbox_crop1 = (0, 0, bbox1[2] - bbox1[0], bbox1[3] - bbox1[1])
                
                if not to_bbox_blot:
                    mask0 = polygon_to_mask(poly0, height, width)
                    mask1 = polygon_to_mask(poly1, height, width)
                else:
                    _, mask0 = polygon_to_bbox_mask(poly0, height, width)
                    _, mask1 = polygon_to_bbox_mask(poly1, height, width)
                
                match_info = {
                    "panel_id0": id0,
                    "panel_id1": id1,
                    'panel_label': panel_label,
                    "poly_coords0": list(poly0.exterior.coords),
                    'bbox_crop0': bbox_crop0,
                    "poly_coords1": list(poly1.exterior.coords),
                    'bbox_crop1': bbox_crop1,
                    'mask0': mask0,
                    'mask1': mask1,
                    'match_result': {
                        'fallback': 'full_bbox',
                        'inliers': 0,
                        'total_matches': 0,
                        'mean_match_score': 0.0,
                    },
                }
                match_results.append(match_info)
            continue

        ##### PREVIOUS LOGIC
        poly0_crop = match_result["overlap_poly_img0"]
        poly1_crop = match_result["overlap_poly_img1"]

        poly0 = translate(poly0_crop, xoff=bbox0[0], yoff=bbox0[1])
        poly1 = translate(poly1_crop, xoff=bbox1[0], yoff=bbox1[1])

        bbox_crop0 = polygon_to_bbox(poly0_crop)
        bbox_crop1 = polygon_to_bbox(poly1_crop)

        if panel_label == 'Blots':
            to_bbox = to_bbox_blot
        elif (panel_label == 'Microscopy') or (panel_label == 'Body Imaging'):
            to_bbox = to_bbox_micro
        else:
            raise ValueError(f"Unknown panel_label: {panel_label}")
        
        if not to_bbox:
            mask0 = polygon_to_mask(poly0, height, width)
            mask1 = polygon_to_mask(poly1, height, width)
        else:
            _, mask0 = polygon_to_bbox_mask(poly0, height, width)
            _, mask1 = polygon_to_bbox_mask(poly1, height, width)

        match_info = {
            "panel_id0": id0,
            "panel_id1": id1,
            'panel_label': panel_label,
            "poly_coords0": list(poly0.exterior.coords),
            'bbox_crop0': bbox_crop0,
            "poly_coords1": list(poly1.exterior.coords),
            'bbox_crop1': bbox_crop1,
            'mask0': mask0,
            'mask1': mask1,
            'match_result': match_result,
        }
        match_results.append(match_info)

    return match_results


def merge_masks_by_max_cliques(
    masks: List[np.ndarray], 
    duplicate_info: List[Dict[str, Any]], 
    verbose: bool = False
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """
    Merge masks that share common panel IDs into maximal cliques.
    
    A clique is a set of panels where ALL possible pairs exist.
    For example: (0,1), (1,2), (0,2) form a complete triangle and get merged.
    But (0,1), (1,2) without (0,2) remain as separate pairs.
    
    Args:
        masks: List of binary masks for duplicate pairs, each shape (H, W)
        duplicate_info: List of match info dictionaries with 'panel_id0' and 'panel_id1'
        verbose: If True, print information when actual merging happens
    
    Returns:
        Tuple of (merged_masks, merged_info):
        - merged_masks: List of merged binary masks
        - merged_info: List of merged info dictionaries containing:
            - 'panel_ids': List of all panel IDs in the clique
            - 'n_panels': Number of panels
            - 'n_pairs': Number of pairs merged
            - 'avg_match_score': Average match score across pairs
            - 'total_inliers': Total number of inlier matches
            - 'all_poly_coords': List of all polygon coordinates
            - 'original_indices': Indices of original pairs that were merged
    """
    # Input validation
    if not isinstance(masks, (list, tuple)) or not isinstance(duplicate_info, (list, tuple)):
        raise TypeError("masks and duplicate_info must be lists or tuples")
    if len(masks) != len(duplicate_info):
        raise ValueError(f"masks and duplicate_info must have same length, got {len(masks)} and {len(duplicate_info)}")
    if not masks:
        return [], []
    
    # Validate duplicate_info structure
    required_keys = ['panel_id0', 'panel_id1']
    for idx, info in enumerate(duplicate_info):
        if not isinstance(info, dict):
            raise TypeError(f"duplicate_info[{idx}] must be a dict, got {type(info)}")
        missing = [key for key in required_keys if key not in info]
        if missing:
            raise KeyError(f"duplicate_info[{idx}] missing required keys: {missing}")
    
    # Build graph and track existing pairs
    graph = defaultdict(set)
    existing_pairs = set()
    pair_to_idx = {}
    
    for idx, info in enumerate(duplicate_info):
        id0 = info['panel_id0']
        id1 = info['panel_id1']
        graph[id0].add(id1)
        graph[id1].add(id0)
        pair = (min(id0, id1), max(id0, id1))
        existing_pairs.add(pair)
        pair_to_idx[pair] = idx
    
    def find_maximal_cliques_bron_kerbosch(R, P, X, cliques):
        """Bron-Kerbosch algorithm to find all maximal cliques."""
        if not P and not X:
            if len(R) > 1:  # Only interested in cliques with 2+ nodes
                cliques.append(R.copy())
            return
        
        for v in list(P):
            neighbors = graph[v]
            find_maximal_cliques_bron_kerbosch(
                R | {v},
                P & neighbors,
                X & neighbors,
                cliques
            )
            P.remove(v)
            X.add(v)
    
    # Find all maximal cliques
    all_nodes = set(graph.keys())
    cliques = []
    find_maximal_cliques_bron_kerbosch(set(), all_nodes, set(), cliques)
    
    # Sort cliques by size (largest first) to prioritize complete groups
    cliques.sort(key=len, reverse=True)
    
    # Track which pairs have been used
    used_pairs = set()
    merged_masks = []
    merged_info = []
    
    # Process cliques with 3+ nodes (actual merges)
    for clique in cliques:
        if len(clique) < 3:
            continue
        
        clique_list = sorted(clique)
        
        # Get all pairs in this clique
        clique_pairs = []
        for i in range(len(clique_list)):
            for j in range(i + 1, len(clique_list)):
                pair = (clique_list[i], clique_list[j])
                if pair in existing_pairs and pair not in used_pairs:
                    clique_pairs.append(pair)
        
        # Skip if no unused pairs
        if not clique_pairs:
            continue
        
        # Mark pairs as used
        for pair in clique_pairs:
            used_pairs.add(pair)
        
        # Get indices and merge masks
        relevant_indices = [pair_to_idx[pair] for pair in clique_pairs]
        
        if verbose:
            print(f"Merging {len(relevant_indices)} pairs into single mask: panels {clique_list}")
            for idx in relevant_indices:
                info = duplicate_info[idx]
                print(f"  - Pair ({info['panel_id0']}, {info['panel_id1']})")
        
        # Merge masks
        merged_mask = np.zeros_like(masks[0], dtype=np.uint8)
        all_polys = []
        total_inliers = 0
        all_match_scores = []
        
        for idx in relevant_indices:
            merged_mask = merged_mask | masks[idx]
            info = duplicate_info[idx]
            all_polys.extend([info['poly_coords0'], info['poly_coords1']])
            total_inliers += info['match_result']['inliers']
            all_match_scores.append(info['match_result']['mean_match_score'])
        
        merged_entry = {
            'panel_ids': clique_list,
            'n_panels': len(clique_list),
            'n_pairs': len(relevant_indices),
            'avg_match_score': np.mean(all_match_scores),
            'total_inliers': total_inliers,
            'all_poly_coords': all_polys,
            'original_indices': relevant_indices
        }
        
        merged_masks.append(merged_mask)
        merged_info.append(merged_entry)
    
    # Add remaining unused pairs as individual masks
    for pair, idx in pair_to_idx.items():
        if pair not in used_pairs:
            info = duplicate_info[idx]
            
            individual_entry = {
                'panel_ids': sorted([info['panel_id0'], info['panel_id1']]),
                'n_panels': 2,
                'n_pairs': 1,
                'avg_match_score': info['match_result']['mean_match_score'],
                'total_inliers': info['match_result']['inliers'],
                'all_poly_coords': [info['poly_coords0'], info['poly_coords1']],
                'original_indices': [idx]
            }
            
            merged_masks.append(masks[idx])
            merged_info.append(individual_entry)
    
    return merged_masks, merged_info