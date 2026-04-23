import numpy as np
from typing import List, Tuple, Dict, Set
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class BlotLane:
    """Represents a single detected blot lane."""
    panel_idx: int      # Index in blot_panels_ids
    bbox: List[float]   # [x0, y0, x1, y1] in panel coordinates
    crop: np.ndarray    # Image crop
    global_idx: int     # Global index across all lanes
    panel_bbox: List[float] = None  # Panel bbox in source image [x0, y0, x1, y1]
    
    @property
    def absolute_bbox(self) -> List[float]:
        """Convert relative bbox to absolute coordinates in source image."""
        if self.panel_bbox is None:
            return self.bbox
        panel_x0, panel_y0, _, _ = self.panel_bbox[-4:]
        x0, y0, x1, y1 = self.bbox
        return [
            panel_x0 + x0,
            panel_y0 + y0,
            panel_x0 + x1,
            panel_y0 + y1
        ]

@dataclass
class BlotMatch:
    """Represents a match between two blot lanes."""
    idx1: int
    idx2: int
    similarity: float
    bbox1_relative: List[float]  # [x0, y0, x1, y1] in panel coordinates
    bbox2_relative: List[float]
    bbox1_absolute: List[float]  # [x0, y0, x1, y1] in source image
    bbox2_absolute: List[float]
    panel_idx1: int
    panel_idx2: int


def boxes_overlap(box1, box2, threshold_pixels=5):
    """Check if boxes overlap by more than threshold pixels."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter > threshold_pixels * threshold_pixels

def build_overlap_groups(lanes: List[BlotLane], overlap_threshold: int = 5) -> List[Set[int]]:
    """
    Group lanes that overlap (from same panel).
    These lanes should never be compared to each other.
    """
    n = len(lanes)
    
    # Build adjacency list of overlapping lanes (same panel only)
    adjacency = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            # Only check overlap if from same panel
            if lanes[i].panel_idx == lanes[j].panel_idx:
                if boxes_overlap(lanes[i].bbox, lanes[j].bbox, overlap_threshold):
                    adjacency[i].add(j)
                    adjacency[j].add(i)
    
    # Find connected components (overlap groups)
    visited = [False] * n
    groups = []
    
    def dfs(idx, group):
        visited[idx] = True
        group.add(idx)
        for neighbor in adjacency[idx]:
            if not visited[neighbor]:
                dfs(neighbor, group)
    
    for i in range(n):
        if not visited[i]:
            group = set()
            dfs(i, group)
            groups.append(group)
    
    return groups

def find_best_match_between_groups(lanes: List[BlotLane],
                                    embeddings: np.ndarray,
                                    overlap_groups: List[Set[int]],
                                    similarity_threshold: float = 0.85,
                                    skip_list: Set[Tuple[int, int]] = None) -> List[BlotMatch]:
    """
    Find pairs of similar lanes, ensuring:
    1. Lanes from the same overlap group are never compared
    2. Lanes from panel pairs in skip_list are never compared
    3. If multiple lanes from group A match multiple lanes from group B,
       only keep the BEST (most similar) pair
    
    Args:
        lanes: List of BlotLane objects
        embeddings: Embeddings for all lanes
        overlap_groups: Groups of overlapping lanes
        similarity_threshold: Threshold for considering lanes as duplicates
        skip_list: Set of normalized panel ID tuples (e.g., {(0, 1)}) to skip matching
    
    Returns:
        List of BlotMatch objects with all coordinate information
    """
    n = len(lanes)
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_emb = embeddings / (norms + 1e-8)
    
    # Compute similarity matrix
    similarity_matrix = np.dot(normalized_emb, normalized_emb.T)
    
    # Create mapping: lane_idx -> overlap_group_idx
    lane_to_group = {}
    for group_idx, group in enumerate(overlap_groups):
        for lane_idx in group:
            lane_to_group[lane_idx] = group_idx
    
    # Normalize skip_list for fast lookup (convert to set of sorted tuples)
    if skip_list is None:
        skip_list = set()
    
    # Find all candidate pairs (not in same overlap group, above threshold, not in skip_list)
    candidate_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            # Skip if in same overlap group
            if lane_to_group[i] == lane_to_group[j]:
                continue
            
            # Skip if panel pair is in skip_list
            panel_i = lanes[i].panel_idx
            panel_j = lanes[j].panel_idx
            panel_pair = tuple(sorted([panel_i, panel_j]))
            if panel_pair in skip_list:
                continue
            
            similarity = similarity_matrix[i, j]
            if similarity >= similarity_threshold:
                candidate_pairs.append((i, j, similarity))
    
    # Group candidates by their overlap groups
    # Key: (group_A_idx, group_B_idx), Value: list of (i, j, score)
    group_pair_matches = defaultdict(list)
    
    for i, j, score in candidate_pairs:
        group_i = lane_to_group[i]
        group_j = lane_to_group[j]
        key = tuple(sorted([group_i, group_j]))
        group_pair_matches[key].append((i, j, score))
    
    # For each group-pair, keep only the BEST match
    best_matches = []
    for (group_a, group_b), matches in group_pair_matches.items():
        # Find the pair with highest similarity
        i, j, score = max(matches, key=lambda x: x[2])
        
        # Create BlotMatch object with all coordinate info
        lane1 = lanes[i]
        lane2 = lanes[j]
        
        match = BlotMatch(
            idx1=i,
            idx2=j,
            similarity=float(score),
            bbox1_relative=lane1.bbox,
            bbox2_relative=lane2.bbox,
            bbox1_absolute=lane1.absolute_bbox,
            bbox2_absolute=lane2.absolute_bbox,
            panel_idx1=lane1.panel_idx,
            panel_idx2=lane2.panel_idx
        )
        best_matches.append(match)
    
    return best_matches

def find_lanes_in_blot_panels(panels: List,
                        blot_panels_ids: List[int],
                        crops_list: List[np.ndarray],
                        segmentator,
                        blot_duplicate_detector,
                        similarity_threshold: float = 0.85,
                        overlap_threshold: int = 5,
                        seg_conf: float = 0.3,
                        seg_iou: float = 0.1,
                        skip_list: List[Tuple[int, int]] = None) -> Dict:
    """
    Complete pipeline for blot duplicate detection.
    
    Args:
        panels: List of panel bounding boxes
        blot_panels_ids: List of panel indices to process
        crops_list: List of panel image crops
        segmentator: Segmentation model for detecting lanes
        blot_duplicate_detector: Embedder model for duplicate detection
        similarity_threshold: Threshold for considering lanes as duplicates
        overlap_threshold: Pixel threshold for considering boxes as overlapping
        seg_conf: Confidence threshold for segmentation
        seg_iou: IoU threshold for segmentation
        bbox_expand_pixels: Amount to expand small boxes in each direction
        small_box_threshold_width: Boxes with width < this value are considered small
        small_box_threshold_height: Boxes with height < this value are considered small
        skip_list: List of tuples (panel_id1, panel_id2) representing panel pairs that should
                   not be considered as matches. E.g., [(0, 1)] means lanes from panel 0 and
                   panel 1 will never be matched, even if they are similar.
    
    Returns dict with:
        - lanes: List of all BlotLane objects
        - embeddings: Embeddings for all lanes
        - overlap_groups: Groups of overlapping lanes (shouldn't be compared)
        - best_matches: List of (idx1, idx2, score) for best matches between groups
        - similarity_matrix: Full similarity matrix
    """
    
    if len(blot_panels_ids) == 0:
        return {
            'lanes': [],
            'embeddings': np.array([]),
            'overlap_groups': [],
            'best_matches': [],
            'similarity_matrix': np.array([])
        }
    
    # Step 1: Segment all panels and collect lanes
    all_lanes = []
    global_idx = 0
    
    for panel_idx in blot_panels_ids:
        img_for_seg = crops_list[panel_idx]
        
        # Get panel bbox in source image
        panel_bbox = panels[panel_idx] if panels else None
        
        # Segment
        results_seg = segmentator.predict(
            img_for_seg, 
            imgsz=640, 
            conf=seg_conf, 
            iou=seg_iou, 
            verbose=False
        )[0]
        
        seg_bboxes = results_seg.boxes.xyxy.tolist() if results_seg.boxes is not None else []
        
        # Create BlotLane objects
        for x0, y0, x1, y1 in seg_bboxes:
            crop = img_for_seg[int(y0):int(y1), int(x0):int(x1)]
            lane = BlotLane(
                panel_idx=panel_idx,
                bbox=[x0, y0, x1, y1],
                crop=crop,
                global_idx=global_idx,
                panel_bbox=panel_bbox
            )
            all_lanes.append(lane)
            global_idx += 1
    
    if len(all_lanes) == 0:
        return {
            'lanes': [],
            'embeddings': np.array([]),
            'overlap_groups': [],
            'best_matches': [],
            'similarity_matrix': np.array([])
        }
    
    # Step 2: Compute embeddings for all lanes
    all_crops = [lane.crop for lane in all_lanes]
    embeddings = blot_duplicate_detector.get_embedding_batch(all_crops).cpu().numpy()
    
    # Step 3: Build overlap groups (lanes that shouldn't be compared)
    overlap_groups = build_overlap_groups(all_lanes, overlap_threshold)
    
    # Step 4: Normalize skip_list (convert to set of sorted tuples for fast lookup)
    skip_set = set()
    if skip_list is not None:
        for pair in skip_list:
            # Normalize: always store as (min, max) tuple
            skip_set.add(tuple(sorted(pair)))
    
    # Step 5: Find best matches between different overlap groups
    best_matches = find_best_match_between_groups(
        all_lanes,
        embeddings,
        overlap_groups,
        similarity_threshold,
        skip_set
    )
    
    return {
        'lanes': all_lanes,
        'embeddings': embeddings,
        'overlap_groups': overlap_groups,
        'best_matches': best_matches,
        'similarity_matrix': embeddings @ embeddings.T
    }


def create_lane_match_masks(img_shape: Tuple[int, int, int],
                       best_matches: List[BlotMatch],
                       lanes: List[BlotLane] = None, 
                       whole_panel_count_thresh=0.5) -> List[np.ndarray]:
    """
    Create binary masks for match pairs, one mask per unique panel pair.
    
    If multiple matches involve the same pair of panels, they are combined
    into a single mask.
    
    If `lanes` is provided:
    - Count how many lanes in each panel are involved in matches.
    - If >50% of lanes in a panel are matched, use the WHOLE panel bbox
      for that panel in the mask instead of individual segment bboxes.
    
    Args:
        img_shape: Shape of source image (height, width, channels)
        best_matches: List of BlotMatch objects
        lanes: Optional list of BlotLane objects (needed to count panel lanes)
        
    Returns:
        List of binary masks, one per unique panel pair, shape (height, width)
    """
    height, width = img_shape[:2]
    
    # If lanes provided, determine which panels should use whole bbox
    panel_use_whole = {}
    if lanes is not None:
        # Count total lanes per panel
        panel_lane_counts = defaultdict(int)
        for lane in lanes:
            panel_lane_counts[lane.panel_idx] += 1
        
        # Count matched lanes per panel
        panel_matched_counts = defaultdict(int)
        for match in best_matches:
            panel_matched_counts[match.panel_idx1] += 1
            panel_matched_counts[match.panel_idx2] += 1
        
        # Determine panels with >50% matched lanes
        for panel_idx, total in panel_lane_counts.items():
            matched = panel_matched_counts.get(panel_idx, 0)
            if matched > whole_panel_count_thresh * total:
                panel_use_whole[panel_idx] = True
    
    # Group matches by unique panel pairs
    panel_pair_masks = {}  # (panel_idx1, panel_idx2) -> mask
    
    for match in best_matches:
        # Normalize panel pair (always lower, higher)
        p1, p2 = match.panel_idx1, match.panel_idx2
        panel_pair = tuple(sorted([p1, p2]))
        
        # Create or retrieve mask for this panel pair
        if panel_pair not in panel_pair_masks:
            panel_pair_masks[panel_pair] = np.zeros((height, width), dtype=np.uint8)
        
        mask = panel_pair_masks[panel_pair]
        
        # Mark first bbox
        if match.panel_idx1 in panel_use_whole:
            # Use whole panel bbox
            for lane in lanes:
                if lane.panel_idx == match.panel_idx1:
                    panel_bbox = lane.panel_bbox
                    if panel_bbox is not None:
                        x0, y0, x1, y1 = panel_bbox[-4:]
                        mask[int(y0):int(y1), int(x0):int(x1)] = 1
                    break
        else:
            # Use segment bbox
            x0, y0, x1, y1 = match.bbox1_absolute
            mask[int(y0):int(y1), int(x0):int(x1)] = 1
        
        # Mark second bbox
        if match.panel_idx2 in panel_use_whole:
            # Use whole panel bbox
            for lane in lanes:
                if lane.panel_idx == match.panel_idx2:
                    panel_bbox = lane.panel_bbox
                    if panel_bbox is not None:
                        x0, y0, x1, y1 = panel_bbox[-4:]
                        mask[int(y0):int(y1), int(x0):int(x1)] = 1
                    break
        else:
            # Use segment bbox
            x0, y0, x1, y1 = match.bbox2_absolute
            mask[int(y0):int(y1), int(x0):int(x1)] = 1
    
    # Return masks in order of panel pairs
    masks = [mask for _, mask in sorted(panel_pair_masks.items())]
    return masks


def visualize_matches_on_image(img: np.ndarray,
                               best_matches: List[BlotMatch],
                               show_labels: bool = True) -> np.ndarray:
    """
    Visualize matches by drawing boxes on the image.
    
    Args:
        img: Source image
        best_matches: List of BlotMatch objects
        show_labels: Whether to show match indices
        
    Returns:
        Image with visualized matches
    """
    import cv2
    
    vis_img = img.copy()
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128)
    ]
    
    for i, match in enumerate(best_matches):
        color = colors[i % len(colors)]
        
        # Draw first bbox
        x0, y0, x1, y1 = [int(v) for v in match.bbox1_absolute]
        cv2.rectangle(vis_img, (x0, y0), (x1, y1), color, 2)
        if show_labels:
            cv2.putText(vis_img, f"M{i}a", (x0, y0-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw second bbox
        x0, y0, x1, y1 = [int(v) for v in match.bbox2_absolute]
        cv2.rectangle(vis_img, (x0, y0), (x1, y1), color, 2)
        if show_labels:
            cv2.putText(vis_img, f"M{i}b", (x0, y0-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw connecting line
        center1 = (
            int((match.bbox1_absolute[0] + match.bbox1_absolute[2]) / 2),
            int((match.bbox1_absolute[1] + match.bbox1_absolute[3]) / 2)
        )
        center2 = (
            int((match.bbox2_absolute[0] + match.bbox2_absolute[2]) / 2),
            int((match.bbox2_absolute[1] + match.bbox2_absolute[3]) / 2)
        )
        cv2.line(vis_img, center1, center2, color, 1)
    
    return vis_img