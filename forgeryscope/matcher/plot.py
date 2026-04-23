import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon


def visualize_duplicate_masks(img, masks, duplicate_info, fallback_masks=None, 
                               fallback_info=None, show_fallbacks=False,
                               save_path=None, dpi=150, show_plot=True):
    """
    Visualize duplicate regions using pre-computed masks and info.
    Handles both regular duplicate_info and merged_info from merge_masks_by_max_cliques.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Source image
    masks : list[np.ndarray]
        List of binary masks for duplicate pairs
    duplicate_info : list[dict]
        List of dictionaries containing match information (regular or merged)
    fallback_masks : list[np.ndarray], optional
        List of binary masks that didn't meet thresholds
    fallback_info : list[dict], optional
        List of dictionaries for fallback matches
    show_fallbacks : bool, default=False
        Whether to show fallback matches with different styling
    save_path : str, optional
        Path to save the figure
    dpi : int, default=150
        DPI for saved figure
    show_plot : bool, default=True
        Whether to display the plot (useful for Jupyter notebooks)
    """
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.imshow(img)
    
    # Color palette for different duplicate pairs
    n_pairs = len(masks) + (len(fallback_masks) if show_fallbacks and fallback_masks else 0)
    colors = plt.cm.rainbow(np.linspace(0, 1, max(n_pairs, 1)))
    
    # Visualize main duplicate pairs
    for idx, (mask, info) in enumerate(zip(masks, duplicate_info)):
        color = colors[idx]
        
        # Create colored overlay from mask
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask > 0] = [*color[:3], 0.3]
        ax.imshow(overlay)
        
        # Handle both regular and merged info structures
        if 'poly1_coords' in info:
            # Regular duplicate pair (from create_duplicate_masks)
            poly_coords_list = [info['poly1_coords'], info['poly2_coords']]
            label = f"Pair {idx+1}"
            detail = f"Inliers: {info['inliers']}, Score: {info['match_score']:.3f}"
        else:
            # Merged group (from merge_masks_by_max_cliques)
            poly_coords_list = info['all_poly_coords']
            panel_ids = info['panel_ids']
            label = f"Group {idx+1} ({info['n_panels']} panels)"
            detail = f"Panels: {panel_ids}\n{info['n_pairs']} pairs, Avg: {info['avg_match_score']:.3f}"
        
        # Draw all polygons
        for poly_coords in poly_coords_list:
            poly_arr = np.array(poly_coords)
            polygon = MplPolygon(poly_arr, fill=False, edgecolor=color, linewidth=3)
            ax.add_patch(polygon)
        
        # Add label at mask centroid
        y_coords, x_coords = np.where(mask > 0)
        if len(x_coords) > 0:
            centroid_x = x_coords.mean()
            centroid_y = y_coords.mean()
            
            ax.text(centroid_x, centroid_y, f"{label}\n{detail}", 
                    fontsize=10, color='white', weight='bold', ha='center',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    # Optionally visualize fallback matches (low confidence)
    if show_fallbacks and fallback_masks and fallback_info:
        start_idx = len(masks)
        for idx, (mask, info) in enumerate(zip(fallback_masks, fallback_info)):
            color_idx = start_idx + idx
            color = colors[color_idx]
            
            # Lighter overlay for fallbacks
            overlay = np.zeros((*mask.shape, 4))
            overlay[mask > 0] = [*color[:3], 0.15]
            ax.imshow(overlay)
            
            # Dashed boundaries for fallbacks
            poly1_coords = np.array(info['poly1_coords'])
            poly2_coords = np.array(info['poly2_coords'])
            
            polygon1 = MplPolygon(poly1_coords, fill=False, edgecolor=color, 
                                  linewidth=2, linestyle='--', alpha=0.6)
            polygon2 = MplPolygon(poly2_coords, fill=False, edgecolor=color, 
                                  linewidth=2, linestyle='--', alpha=0.6)
            ax.add_patch(polygon1)
            ax.add_patch(polygon2)
            
            # Label fallbacks
            poly1 = Polygon(poly1_coords)
            label = f"Low conf {idx+1}"
            ax.text(poly1.centroid.x, poly1.centroid.y, label, 
                    fontsize=9, color='white', style='italic', ha='center',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.6))
    
    title = f"Duplicate Regions ({len(masks)} groups"
    if show_fallbacks and fallback_masks:
        title += f", {len(fallback_masks)} low-confidence"
    title += ")"
    
    ax.set_title(title, fontsize=16, weight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Validate and convert unsupported formats
        supported_formats = {'eps', 'jpeg', 'jpg', 'pdf', 'pgf', 'png', 'ps', 'raw', 'rgba', 'svg', 'svgz', 'tif', 'tiff', 'webp'}
        _, ext = os.path.splitext(save_path)
        file_format = ext.lstrip('.').lower()
        
        if file_format not in supported_formats:
            # Convert unsupported formats (e.g., jp2) to png
            orig_path = save_path
            save_path = os.path.splitext(save_path)[0] + '.png'
            print(f"Format '{file_format}' not supported. Converting to PNG: {save_path}")
        
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        print(f"Figure saved to: {save_path}")
    
    # Display plot if requested (useful for Jupyter notebooks)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    # Return summary for reference
    summary = {
        'n_duplicates': len(masks),
        'n_fallbacks': len(fallback_masks) if fallback_masks else 0,
        'duplicate_pairs': [
            {
                'pair_id': i + 1,
                'info': info
            }
            for i, info in enumerate(duplicate_info)
        ]
    }
    
    return summary