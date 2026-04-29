# Forgeryscope

Forgeryscope is a Python package for scientific image forgery detection. It
contains tools for panel detection, embedding, and image/panel matching.

This is a simplified and refactored version of my winning solution for the
Kaggle [Scientific Image Forgery Detection](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection)
competition.

For the full competition approach and design notes, see
[SOLUTION.md](SOLUTION.md).

## Features

- **Panel detection** with YOLO-based extractors
- **Image embeddings** with PyTorch/timm checkpoints
- **Panel matching** with LightGlue, SIFT, ALIKED, and geometry helpers
- **On-demand model download** from GitHub Releases

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install Forgeryscope from PyPI:

```bash
pip install forgeryscope
```

To install the latest version directly from GitHub:

```bash
pip install git+https://github.com/vlad3996/forgeryscope.git
```

If you work from a local clone:

```bash
git clone https://github.com/vlad3996/forgeryscope.git
cd forgeryscope
pip install -e .
```

## Model Weights

Model weights are not stored in git and are not bundled into the Python package.
They are downloaded on first use from the Forgeryscope GitHub Release and
cached locally.

Forgeryscope resolves these model names through `forgeryscope.model_zoo`:

- `yolo_panel_extractor.pt`
- `yolo_lane_extractor.pt`
- `aliked_wblot.pth`
- `wblot_duplicate_embedder.ckpt`
- `wblot_overlap_embedder.ckpt`
- `wblot_lane_embedder.ckpt`
- `micro_overlap_embedder.ckpt`

To override the default model release location:

```bash
export FORGERYSCOPE_MODEL_BASE_URL="https://github.com/vlad3996/forgeryscope/releases/download/models-v1"
```

The first run downloads each requested file to:

```bash
~/.cache/forgeryscope
```

To use a different cache directory:

```bash
export FORGERYSCOPE_CACHE_DIR="/path/to/cache"
```

## Quick Start

```python
from ultralytics import YOLO
import numpy as np
import pandas as pd
from PIL import Image

from forgeryscope import Embedder, PanelExtractor, get_model_path, load_aliked_wblot_weights
from forgeryscope.matcher.lightglue import LightGlueOverlap, create_duplicate_masks, merge_masks_by_max_cliques
from forgeryscope.matcher.geometry import get_intersections
from forgeryscope.matcher.plot import visualize_duplicate_masks
from forgeryscope.matcher.lane import find_lanes_in_blot_panels, create_lane_match_masks

DEVICE = "cuda"
PRINT_MODEL_DEFINITION = False
VERBOSE = False

panel_extractor = PanelExtractor(
    weights_path="yolo_panel_extractor",
    device=DEVICE,
    conf_threshold=0.7,
    iou_threshold=0.4,
    verbose=PRINT_MODEL_DEFINITION,
)
panel_extractor.EXCLUDED_LABELS = {"Graphs", "Flow Cytometry", "Body Imaging"}

lane_extractor = YOLO(get_model_path("yolo_lane_extractor"))

wblot_duplicate_embedder = Embedder("wblot_duplicate_embedder", device=DEVICE, verbose=PRINT_MODEL_DEFINITION)
wblot_overlap_embedder = Embedder("wblot_overlap_embedder", device=DEVICE, verbose=PRINT_MODEL_DEFINITION)
wblot_lane_embedder = Embedder("wblot_lane_embedder", device=DEVICE, verbose=PRINT_MODEL_DEFINITION)
micro_overlap_embedder = Embedder("micro_overlap_embedder", device=DEVICE, verbose=PRINT_MODEL_DEFINITION)

matcher_micro = LightGlueOverlap(
    max_keypoints=4096,
    matcher_features="sift",
    device=DEVICE,
    depth_confidence=0.9,
    width_confidence=0.9,
    verbose=PRINT_MODEL_DEFINITION,
)

matcher_blot = LightGlueOverlap(
    max_keypoints=512,
    matcher_features="aliked",
    device=DEVICE,
    depth_confidence=-1,
    width_confidence=-1,
    estimator_method="MAGSAC",
    reprojThreshold=3.0,
    estimator_confidence=0.9999,
    estimator_maxIters=5000,
    estimator_refineIters=10,
    verbose=PRINT_MODEL_DEFINITION,
)
load_aliked_wblot_weights(matcher_blot)
```

Optional embedding sanity check:

```python
img1 = np.array(Image.open("/path/to/wblot_sample.png").convert("RGB"))
img2 = np.array(Image.open("/path/to/wblot_sample_sub2.png").convert("RGB"))

print("wblot_duplicate_embedder:", wblot_duplicate_embedder.compare(img1, img2))
print("wblot_overlap_embedder:", wblot_overlap_embedder.compare(img1, img2))
print("micro_overlap_embedder:", micro_overlap_embedder.compare(img1, img2))
```

Single-image example:

```python
MATCH_SCORE_THRESHOLD = 0.73
INLIER_THRESHOLD = 8
MATCH_FILTER_STR = "mean_match_score"

WBLOT_DUP_SCORE_THRESH = 0.84
MICROSCOPY_EMB_THRESH = 0.58
MICRO_DUP_SCORE_THRESH = 0.85
WBLOT_OVERLAP_THRESHOLD = 0.85
SEG_SIM_THRESH = 0.65

def find_similar_panel_pairs(panel_ids, crops, embedder, threshold, label):
    embeddings = embedder.get_embedding_batch(crops).cpu()
    return [
        (label, score, panel_ids[i], panel_ids[j])
        for i, j, score in Embedder.find_similar_pairs(embeddings, threshold=threshold)
    ]


image_path = "/path/to/image.png"
img = PanelExtractor._load_image(image_path)
panels = panel_extractor.extract_panels(img)
crops_list = PanelExtractor.crop_panels(img, panels)
intersections = get_intersections(panels, margin=10)

blot_ids = [i for i, panel in enumerate(panels) if panel[0] == "Blots"]
micro_ids = [i for i, panel in enumerate(panels) if panel[0] == "Microscopy"]

similar_pairs = []
similar_pairs_blot = []
if blot_ids:
    blot_crops = PanelExtractor.crop_panels(img, [panels[i] for i in blot_ids])
    blot_overlap_pairs = find_similar_panel_pairs(
        blot_ids,
        blot_crops,
        wblot_overlap_embedder,
        WBLOT_OVERLAP_THRESHOLD,
        "Blots",
    )
    blot_duplicate_pairs = find_similar_panel_pairs(
        blot_ids,
        blot_crops,
        wblot_duplicate_embedder,
        WBLOT_DUP_SCORE_THRESH,
        "Blots",
    )

    best_blot_scores = {}
    for label, score, i, j in blot_overlap_pairs + blot_duplicate_pairs:
        key = tuple(sorted((i, j)))
        if key not in best_blot_scores or score > best_blot_scores[key]:
            best_blot_scores[key] = score

    similar_pairs_blot = [
        ("Blots", score, i, j)
        for (i, j), score in best_blot_scores.items()
    ]
    similar_pairs.extend(similar_pairs_blot)

if micro_ids:
    micro_crops = PanelExtractor.crop_panels(img, [panels[i] for i in micro_ids])
    similar_pairs.extend(
        find_similar_panel_pairs(
            micro_ids,
            micro_crops,
            micro_overlap_embedder,
            MICROSCOPY_EMB_THRESH,
            "Microscopy",
        )
    )

similar_pairs = [
    (label, score, i, j)
    for label, score, i, j in similar_pairs
    if (i, j) not in intersections and (j, i) not in intersections
]

clf_predicts = pd.DataFrame(similar_pairs, columns=["label", "score", "idx1", "idx2"])
match_results = create_duplicate_masks(
    img,
    panels,
    crops_list,
    clf_predicts,
    matcher_micro,
    matcher_blot,
    to_bbox_micro=False,
    to_bbox_blot=True,
    fallback_for_wblot=True,
    test_transforms_blot=False,
    test_transforms_micro=True,
)

pred_masks, duplicate_info = [], []
for info in match_results:
    label = info["panel_label"]
    match_result = info["match_result"]
    inliers = match_result["inliers"]
    matcher_score = match_result[MATCH_FILTER_STR]

    if label != "Blots":
        if inliers < INLIER_THRESHOLD or matcher_score < MATCH_SCORE_THRESHOLD:
            continue

    pred_masks.append((info["mask0"] | info["mask1"]).astype(np.uint8))
    duplicate_info.append(info)

mask_matcher, merged_info = merge_masks_by_max_cliques(
    pred_masks,
    duplicate_info,
    verbose=VERBOSE,
)

if blot_ids and not similar_pairs_blot:
    lane_match_result = find_lanes_in_blot_panels(
        panels=panels,
        blot_panels_ids=blot_ids,
        crops_list=crops_list,
        segmentator=lane_extractor,
        blot_duplicate_detector=wblot_lane_embedder,
        similarity_threshold=SEG_SIM_THRESH,
        overlap_threshold=5,
    )
    if lane_match_result:
        mask_lanes = create_lane_match_masks(
            img.shape,
            lane_match_result["best_matches"],
            lanes=lane_match_result["lanes"],
        )
        mask_matcher += mask_lanes

annotation = "authentic" if not mask_matcher else mask_matcher
visualize_duplicate_masks(img, mask_matcher, merged_info, show_fallbacks=False)
```

You can also pass a local checkpoint path instead of a model name:

```python
embedder = Embedder("/path/to/wblot_duplicate_embedder.ckpt", device="cuda", verbose=False)
```

## Available Model Names

- `yolo_panel_extractor`
- `yolo_lane_extractor`
- `aliked_wblot`
- `wblot_duplicate_embedder`
- `wblot_overlap_embedder`
- `wblot_lane_embedder`
- `micro_overlap_embedder`

## Maintainer Notes

Keep `checkpoints/` ignored by git. Large `.pt`, `.pth`, and `.ckpt` files
should live in GitHub Releases, not in normal repository history.

When a checkpoint changes:

1. Upload the new file to a new GitHub Release tag.
2. Update `FORGERYSCOPE_MODEL_BASE_URL` to the new release URL.
3. Update the SHA256 value in `forgeryscope/model_zoo.py`.

## Requirements

- Python >= 3.9
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- OpenCV >= 4.5.0
- Dependencies listed in `pyproject.toml`

## License

The original Forgeryscope source code in this repository is licensed under the
MIT License. See [LICENSE](LICENSE).

Third-party dependencies and model weights may be governed by separate terms.
In particular, Ultralytics YOLO code and trained YOLO models are licensed by
Ultralytics under AGPL-3.0 by default, with separate Enterprise licensing
available from Ultralytics for use cases that cannot comply with AGPL-3.0.
This applies to the YOLO-based detector weights distributed for this project.

## Author

Uladzislau Leketush (vlad.leketush@gmail.com)
