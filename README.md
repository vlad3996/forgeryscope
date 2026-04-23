# Forgeryscope

A Python package for panel detection, embedding, and matching using deep learning.

A simplified and refactored version of my winning solution for the Kaggle [Scientific Image Forgery Detection](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection) competition.

## Features

- **Panel Detection** - Extract panels from images using YOLO-based detection
- **Image Embedding** - Generate embeddings using PyTorch models
- **Panel Matching** - Match corresponding panels using LightGlue

## Installation

```bash
pip install forgeryscope
```

## Quick Start

```python
from forgeryscope import PanelExtractor, Embedder, LightGlueOverlap

# Extract panels from an image
extractor = PanelExtractor()
panels = extractor.extract("image.jpg")

# Generate embeddings
embedder = Embedder()
embeddings = embedder.embed(panels)

# Match panels across images
matcher = LightGlueOverlap()
matches = matcher.match(embeddings1, embeddings2)
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- OpenCV >= 4.5.0
- And other dependencies listed in `pyproject.toml`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Uladzislau Leketush (vlad.leketush@gmail.com)
