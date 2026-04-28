from typing import Union, List, Tuple
import numpy as np
from PIL import Image
from ultralytics import YOLO
from forgeryscope.model_zoo import get_model_path


class PanelExtractor:   
    EXCLUDED_LABELS = {"Graphs", "Flow Cytometry"}
    
    def __init__(
        self, 
        weights_path: str = "yolo_panel_extractor", 
        device: str = 'cpu', 
        img_size: int = 640, 
        conf_threshold: float = 0.3, 
        iou_threshold: float = 0.4, 
        min_crop_side: int = 5,
        model_base_url: str = None,
        cache_dir: str = None,
    ):
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_crop_side = min_crop_side
        self.img_size = img_size
        
        weights_path = get_model_path(weights_path, cache_dir=cache_dir, base_url=model_base_url)
        self.model = YOLO(weights_path)
        self.model.to(device)
        self.names = self.model.names
        
        print(f"Detector loaded successfully from {weights_path}. Classes: {self.names}")
        print(f"Size: {self.img_size} | conf thresh: {self.conf_threshold}, IOU: {self.iou_threshold}, min crop side: {self.min_crop_side} pixels")

    def extract_panels(
        self, 
        img: Union[str, Image.Image, np.ndarray], 
        with_id: bool = False
    ) -> List[Union[Tuple[str, float, float, float, float, float], Tuple[int, str, float, float, float, float, float]]]:
        """
        Extract panels from an image.
        
        Args:
            img: Can be a file path (str), PIL Image, or numpy array
            with_id: If True, returns tuples with index: (idx, label, confidence, x0, y0, x1, y1).
                     If False, returns: (label, confidence, x0, y0, x1, y1)
            
        Returns:
            List of tuples. Format depends on with_id:
            - If with_id=False: (label, confidence, x0, y0, x1, y1)
            - If with_id=True: (idx, label, confidence, x0, y0, x1, y1)
        """
        if isinstance(img, str):
            img = self._load_image(img)
        elif isinstance(img, Image.Image):
            img = np.array(img)
        elif not isinstance(img, np.ndarray):
            raise TypeError("img must be np.ndarray, PIL.Image, or file path")
        
        results = self.model.predict(
            img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            verbose=False
        )
        
        panels = self._parse_predictions(results[0], with_id)
        return panels
        
    def _parse_predictions(
        self, 
        result, 
        with_id: bool = False, 
        round_to: int = 3
    ) -> List[Union[Tuple[str, float, float, float, float, float], Tuple[int, str, float, float, float, float, float]]]:
        """Parse YOLOv11 results into panel format."""
        panels = []
        
        boxes = result.boxes
        
        if boxes is None or len(boxes) == 0:
            return panels
        
        save_idx = 0
        for box in boxes:
            # Extract box coordinates (xyxy format)
            x0, y0, x1, y1 = box.xyxy[0].cpu().numpy()
            
            # Extract confidence and class
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = self.names[cls]
            
            # Skip excluded labels
            if label in self.EXCLUDED_LABELS:
                continue
            
            # Filter by minimum crop size
            if x1 - x0 < self.min_crop_side or y1 - y0 < self.min_crop_side:
                continue
            
            confidence = round(conf, round_to)
            if not with_id:
                panels.append((label, confidence, float(x0), float(y0), float(x1), float(y1)))
            else:
                panels.append((save_idx, label, confidence, float(x0), float(y0), float(x1), float(y1)))
            save_idx += 1
        
        return panels

    @staticmethod
    def _load_image(image_path: str) -> np.ndarray:
        """Load image from file path."""
        img = Image.open(image_path).convert("RGB")
        return np.array(img)
    
    @staticmethod
    def crop_panels(
        img: np.ndarray, 
        panels: List[Union[Tuple[str, float, float, float, float, float], Tuple[int, str, float, float, float, float, float]]]
    ) -> List[np.ndarray]:
        """
        Crop panels from image based on detection results.
        
        Args:
            img: numpy array of the image
            panels: List of tuples. Can be either:
                   - (label, conf, x0, y0, x1, y1) - 6 elements
                   - (idx, label, conf, x0, y0, x1, y1) - 7 elements
            
        Returns:
            List of cropped image arrays
        """
        if panels is None or not len(panels):
            return []
        

        h, w = img.shape[:2]
        crops = []
        
        for panel in panels:
            if len(panel) == 6:
                label, conf, x0, y0, x1, y1 = panel
            elif len(panel) == 7:
                idx, label, conf, x0, y0, x1, y1 = panel
            else:
                raise ValueError(f"Panel tuple must have 6 or 7 elements, got {len(panel)}")
            
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(w, int(x1)), min(h, int(y1))
            
            if x1 <= x0 or y1 <= y0: 
                continue
                
            crop = img[int(y0):int(y1), int(x0):int(x1)]
            crops.append(crop)
            
        return crops
    
    @staticmethod
    def visualize_detections(
        img: Union[str, Image.Image, np.ndarray], 
        panels: List[Union[Tuple[str, float, float, float, float, float], Tuple[int, str, float, float, float, float, float]]], 
        figsize: Tuple[int, int] = (15, 10), 
        show_confidence: bool = True, 
        line_width: int = 3, 
        font_size: int = 12
    ):
        """
        Visualize detected panels on the source image.
        
        Args:
            img: numpy array or PIL Image or file path
            panels: List of tuples. Can be either:
                   - (label, conf, x0, y0, x1, y1) - 6 elements
                   - (idx, label, conf, x0, y0, x1, y1) - 7 elements
            figsize: Figure size (width, height)
            show_confidence: Whether to show confidence scores in labels
            line_width: Width of bounding box lines
            font_size: Font size for labels
            
        Returns:
            matplotlib.figure.Figure: matplotlib figure object
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        # Load image if path provided
        if isinstance(img, str):
            img = np.array(Image.open(img).convert("RGB"))
        elif isinstance(img, Image.Image):
            img = np.array(img)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(img)
        
        # Color map for different classes
        colors = plt.cm.Set3(np.linspace(0, 1, 12))
        label_colors = {}
        
        # Draw bounding boxes
        for panel_idx, panel in enumerate(panels):
            # Handle both 6-element and 7-element panel formats
            if len(panel) == 6:
                label, conf, x0, y0, x1, y1 = panel
                display_idx = panel_idx
            elif len(panel) == 7:
                display_idx, label, conf, x0, y0, x1, y1 = panel
            else:
                raise ValueError(f"Panel tuple must have 6 or 7 elements, got {len(panel)}")
            # Assign consistent color per label
            if label not in label_colors:
                label_colors[label] = colors[len(label_colors) % len(colors)]
            
            color = label_colors[label]
            
            # Draw rectangle
            rect = patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=line_width,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label with confidence
            if show_confidence:
                text = f"#{display_idx} {label} {conf:.2f}"
            else:
                text = f"#{display_idx} {label}"
            
            # Add text background for better visibility
            ax.text(
                x0, y0 - 5,
                text,
                fontsize=font_size,
                color='white',
                weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8)
            )
        
        ax.axis('off')
        ax.set_title(f'Detected Panels: {len(panels)} detections', 
                    fontsize=font_size + 4, weight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def visualize_with_crops(
        img: Union[str, Image.Image, np.ndarray], 
        panels: List[Union[Tuple[str, float, float, float, float, float], Tuple[int, str, float, float, float, float, float]]], 
        max_crops: int = 10, 
        figsize: Tuple[int, int] = (20, 12)
    ):
        """
        Visualize both the full image with detections and individual crops.
        
        Args:
            img: numpy array or PIL Image or file path
            panels: List of tuples. Can be either:
                   - (label, conf, x0, y0, x1, y1) - 6 elements
                   - (idx, label, conf, x0, y0, x1, y1) - 7 elements
            max_crops: Maximum number of crops to display
            figsize: Figure size (width, height)
            
        Returns:
            matplotlib.figure.Figure: matplotlib figure object
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        # Load image if path provided
        if isinstance(img, str):
            img_array = np.array(Image.open(img).convert("RGB"))
        elif isinstance(img, Image.Image):
            img_array = np.array(img)
        else:
            img_array = img
        
        # Get crops
        crops = PanelExtractor.crop_panels(img_array, panels)
        n_crops = min(len(crops), max_crops)
        
        # Create subplot layout
        n_cols = min(5, n_crops)
        n_rows = 1 + (n_crops + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.3)
        
        # First row: full image with detections
        ax_main = fig.add_subplot(gs[0, :])
        ax_main.imshow(img_array)
        
        # Draw boxes on main image
        colors = plt.cm.Set3(np.linspace(0, 1, 12))
        label_colors = {}
        
        for panel_idx, panel in enumerate(panels[:max_crops]):
            # Handle both 6-element and 7-element panel formats
            if len(panel) == 6:
                label, conf, x0, y0, x1, y1 = panel
                display_idx = panel_idx
            elif len(panel) == 7:
                display_idx, label, conf, x0, y0, x1, y1 = panel
            else:
                raise ValueError(f"Panel tuple must have 6 or 7 elements, got {len(panel)}")
            if label not in label_colors:
                label_colors[label] = colors[len(label_colors) % len(colors)]
            
            color = label_colors[label]
            rect = patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            ax_main.add_patch(rect)
            
            # Add numbered label
            ax_main.text(
                x0, y0 - 5,
                f"{display_idx}",
                fontsize=14,
                color='white',
                weight='bold',
                bbox=dict(boxstyle='circle,pad=0.3', facecolor=color, alpha=0.9)
            )
        
        ax_main.axis('off')
        ax_main.set_title(f'Source Image with {len(panels)} Detections', 
                         fontsize=16, weight='bold')
        
        # Remaining rows: individual crops
        for crop_idx, crop in enumerate(crops[:max_crops]):
            row = 1 + crop_idx // n_cols
            col = crop_idx % n_cols
            
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(crop)
            ax.axis('off')
            
            # Handle both 6-element and 7-element panel formats
            panel = panels[crop_idx]
            if len(panel) == 6:
                label, conf, x0, y0, x1, y1 = panel
                display_idx = crop_idx
            elif len(panel) == 7:
                display_idx, label, conf, x0, y0, x1, y1 = panel
            else:
                raise ValueError(f"Panel tuple must have 6 or 7 elements, got {len(panel)}")
            
            ax.set_title(f"{display_idx}. {label}\n{conf:.3f}", 
                        fontsize=10, weight='bold')
        
        plt.suptitle('Panel Detection Results', fontsize=18, weight='bold', y=0.98)
        return fig
