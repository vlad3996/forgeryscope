"""
Geometry utility functions for polygon and bounding box operations.

This module provides functions for converting between polygons, bounding boxes,
and masks, as well as computing intersections and IoU metrics.
"""
from typing import Union, List, Tuple, Any
import numpy as np
from shapely.geometry import Polygon, box
from skimage.draw import polygon as sk_polygon, rectangle


def clip_polygon_to_image(poly: Polygon, H: int, W: int) -> Polygon:
    """
    Clip a polygon to image boundaries.
    
    Args:
        poly: Shapely Polygon to clip
        H: Image height
        W: Image width
    
    Returns:
        Shapely Polygon clipped to image bounds [0, 0, W, H]
    """
    img_box = box(0, 0, W, H)
    return poly.intersection(img_box)


def polygon_to_bbox_mask(
    polygon: Polygon, 
    height: int, 
    width: int, 
    padding: int = 0
) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
    """
    Convert polygon to bounding box and create binary mask.
    
    Args:
        polygon: Shapely Polygon
        height: Image height
        width: Image width
        padding: Padding to add around bbox (default: 0)
    
    Returns:
        Tuple of (bbox, mask):
        - bbox: (x_min, y_min, x_max, y_max) as integers
        - mask: Binary mask of shape (height, width), dtype uint8
    """
    bbox = polygon_to_bbox(polygon, padding=padding)
    mask = bbox_to_mask(bbox, height, width)
    return bbox, mask


def polygon_to_mask(poly: Polygon, height: int, width: int) -> np.ndarray:
    """
    Converts a shapely Polygon into a binary mask of size (H, W)
    
    Args:
        poly: Shapely Polygon
        height: Image height
        width: Image width
    
    Returns:
        Binary mask of shape (height, width), dtype uint8
    """
    x, y = poly.exterior.xy
    rr, cc = sk_polygon(y, x, shape=(height, width))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[rr, cc] = 1
    return mask


def polygon_to_bbox(
    polygon: Union[Polygon, List[Tuple[float, float]]], 
    padding: int = 0
) -> Tuple[int, int, int, int]:
    """
    Extract bounding box from polygon.
    
    Args:
        polygon: Shapely Polygon or list of (x, y) coordinate tuples
        padding: Padding to add around bbox (default: 0)
    
    Returns:
        Tuple of (x_min, y_min, x_max, y_max) as integers
    """
    # If Shapely polygon, extract points
    if hasattr(polygon, "exterior"):
        coords = list(polygon.exterior.coords)
    else:
        coords = polygon

    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]

    x_min = min(xs) - padding
    x_max = max(xs) + padding
    y_min = min(ys) - padding
    y_max = max(ys) + padding

    return int(x_min), int(y_min), int(x_max), int(y_max)


def bbox_to_mask(
    bbox: Union[Tuple[int, int, int, int], List[int]], 
    height: int, 
    width: int
) -> np.ndarray:
    """
    Convert bounding box to binary mask.
    
    Args:
        bbox: Bounding box as tuple or list (x_min, y_min, x_max, y_max)
        height: Image height
        width: Image width
    
    Returns:
        Binary mask of shape (height, width), dtype uint8, with 1s inside bbox
    """
    x_min, y_min, x_max, y_max = bbox
    
    # Ensure integer indexing
    x_min, y_min = int(x_min), int(y_min)
    x_max, y_max = int(x_max), int(y_max)

    # Rectangle coords use start=(row, col), end=(row, col)
    start = (y_min, x_min)
    extent = (y_max - y_min, x_max - x_min)

    mask = np.zeros((height, width), dtype=np.uint8)
    
    rr, cc = rectangle(start=start, extent=extent, shape=mask.shape)
    mask[rr, cc] = 1

    return mask


def iou(
    box1: Union[Tuple[float, float, float, float], List[float]], 
    box2: Union[Tuple[float, float, float, float], List[float]]
) -> float:
    """
    Compute Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1: First bounding box (x_min, y_min, x_max, y_max)
        box2: Second bounding box (x_min, y_min, x_max, y_max)
    
    Returns:
        IoU value in [0, 1], or 0.0 if boxes don't overlap
    """
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2

    xi1 = max(x1, x1b)
    yi1 = max(y1, y1b)
    xi2 = min(x2, x2b)
    yi2 = min(y2, y2b)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    inter = (xi2 - xi1) * (yi2 - yi1)
    a1 = (x2 - x1) * (y2 - y1)
    a2 = (x2b - x1b) * (y2b - y1b)
    union = a1 + a2 - inter

    return inter / union if union > 0 else 0.0


def get_intersections(
    panels: List[Tuple[Any, ...]], 
    margin: float = 0
) -> set:
    """
    Find intersecting panels with a margin.
    
    Args:
        panels: List of panel tuples, each ending with (x0, y0, x1, y1) coordinates
        margin: Allowed intersection in pixels (default: 0)
        
    Returns:
        Set of tuples (i, j) where panels[i] intersects with panels[j] 
        by more than the margin. Only includes pairs with i < j.
    """
    intersections = set()
    
    for i in range(len(panels)):
        x0_i, y0_i, x1_i, y1_i = panels[i][-4:]
        for j in range(i + 1, len(panels)):
            x0_j, y0_j, x1_j, y1_j = panels[j][-4:]

            # Calculate overlap in x and y
            overlap_x = max(0, min(x1_i, x1_j) - max(x0_i, x0_j))
            overlap_y = max(0, min(y1_i, y1_j) - max(y0_i, y0_j))

            if overlap_x > margin and overlap_y > margin:
                intersections.add((i, j))
    
    return intersections


def keep_largest_in_intersections(
    panels: List[Tuple[Any, ...]], 
    iou_threshold: float = 0.25, 
    containment_threshold: float = 0.25
) -> List[Tuple[Any, ...]]:
    """
    For any group of overlapping or containing boxes, keep only the largest.
    
    Removes panels that overlap significantly with larger panels, keeping
    only the largest panel in each overlapping cluster.
    
    Args:
        panels: List of panel tuples, each as (label, conf, x0, y0, x1, y1)
        iou_threshold: IoU threshold for considering boxes as overlapping (default: 0.25)
        containment_threshold: Containment ratio threshold (default: 0.25)
        
    Returns:
        Filtered list of panels with overlapping ones removed, keeping largest in each cluster
    """

    def box_area(box):
        x0, y0, x1, y1 = box
        return max(0, x1 - x0) * max(0, y1 - y0)

    def intersection_area(a, b):
        x0 = max(a[0], b[0])
        y0 = max(a[1], b[1])
        x1 = min(a[2], b[2])
        y1 = min(a[3], b[3])
        if x1 <= x0 or y1 <= y0:
            return 0
        return (x1 - x0) * (y1 - y0)

    def iou(a, b):
        inter = intersection_area(a, b)
        if inter == 0:
            return 0.0
        ua = box_area(a)
        ub = box_area(b)
        return inter / (ua + ub - inter)

    # containment = intersection / area(smaller_box)
    def containment(a, b):
        inter = intersection_area(a, b)
        if inter == 0:
            return 0.0
        return inter / min(box_area(a), box_area(b))

    n = len(panels)
    if n <= 1:
        return panels

    boxes = [p[2:6] for p in panels]

    # Build adjacency graph
    graph = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (
                iou(boxes[i], boxes[j]) > iou_threshold
                or containment(boxes[i], boxes[j]) > containment_threshold
            ):
                graph[i].append(j)
                graph[j].append(i)

    # Find connected components
    visited = [False] * n
    clusters = []

    def dfs(node, group):
        visited[node] = True
        group.append(node)
        for nei in graph[node]:
            if not visited[nei]:
                dfs(nei, group)

    for i in range(n):
        if not visited[i]:
            group = []
            dfs(i, group)
            clusters.append(group)

    # Keep largest in each cluster
    kept = []
    for cluster in clusters:
        if len(cluster) == 1:
            kept.append(panels[cluster[0]])
            continue

        largest = max(cluster, key=lambda idx: box_area(boxes[idx]))
        kept.append(panels[largest])

    return kept

