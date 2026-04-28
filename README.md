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

## Linux Installation

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

The package is available on PyPI:
[https://pypi.org/project/forgeryscope/](https://pypi.org/project/forgeryscope/)

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
They are downloaded on first use and cached locally.

Upload these files as assets to a GitHub Release, for example release tag
`models-v1`:

- `yolo_panel_extractor.pt`
- `yolo_lane_extractor.pt`
- `aliked_wblot.pth`
- `wblot_duplicate_embedder.ckpt`
- `wblot_overlap_embedder.ckpt`
- `wblot_lane_embedder.ckpt`
- `micro_overlap_embedder.ckpt`

Then set the model URL in your Linux shell:

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

from forgeryscope import (
    Embedder,
    LightGlueOverlap,
    PanelExtractor,
    get_model_path,
    load_aliked_wblot_weights,
)

device = "cuda"

panel_extractor = PanelExtractor(
    weights_path="yolo_panel_extractor",
    device=device,
    conf_threshold=0.7,
    iou_threshold=0.4,
    verbose=False,
)
panel_extractor.EXCLUDED_LABELS = {"Graphs", "Flow Cytometry", "Body Imaging"}

lane_extractor = YOLO(get_model_path("yolo_lane_extractor"))

matcher_micro = LightGlueOverlap(
    max_keypoints=4096,
    matcher_features="sift",
    device=device,
    depth_confidence=0.9,
    width_confidence=0.9,
    verbose=False,
)

matcher_blot = LightGlueOverlap(
    max_keypoints=512,
    matcher_features="aliked",
    device=device,
    depth_confidence=-1,
    width_confidence=-1,
    estimator_method="MAGSAC",
    reprojThreshold=3.0,
    estimator_confidence=0.9999,
    estimator_maxIters=5000,
    estimator_refineIters=10,
    verbose=False,
)
load_aliked_wblot_weights(matcher_blot)

wblot_duplicate_embedder = Embedder("wblot_duplicate_embedder", device=device, verbose=False)
wblot_overlap_embedder = Embedder("wblot_overlap_embedder", device=device, verbose=False)
wblot_lane_embedder = Embedder("wblot_lane_embedder", device=device, verbose=False)
micro_overlap_embedder = Embedder("micro_overlap_embedder", device=device, verbose=False)
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

## Publishing Notes

Keep `checkpoints/` ignored by git. Large `.pt`, `.pth`, and `.ckpt` files
should live in GitHub Releases, not in normal repository history.

When a checkpoint changes:

1. Upload the new file to a new GitHub Release tag.
2. Update `FORGERYSCOPE_MODEL_BASE_URL` to the new release URL.
3. Update the SHA256 value in `forgeryscope/model_zoo.py`.

## Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- OpenCV >= 4.5.0
- Dependencies listed in `pyproject.toml`

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Author

Uladzislau Leketush (vlad.leketush@gmail.com)
