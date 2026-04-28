import hashlib
import os
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse
from urllib.request import urlopen

import torch
from tqdm.auto import tqdm


DEFAULT_MODEL_BASE_URL = "https://github.com/vlad3996/forgeryscope/releases/download/models-v1"


MODEL_SPECS: Dict[str, Dict[str, object]] = {
    "yolo_panel_extractor": {
        "filename": "yolo_panel_extractor.pt",
        "sha256": "f686674cfb06e73c0e95791cd5ba73823a2cccc918b0bd67f6418e41f88d057f",
    },
    "yolo_lane_extractor": {
        "filename": "yolo_lane_extractor.pt",
        "sha256": "0061376bcb8895707ac8d052698c21849ee151c6c7bfc6d00de0939422571712",
    },
    "aliked_wblot": {
        "filename": "aliked_wblot.pth",
        "sha256": "e9ddeb9e87b96555a6db4a4fbfb5b99b31c36371bf4171765dce8f1f9ef95022",
    },
    "wblot_duplicate_embedder": {
        "filename": "wblot_duplicate_embedder.ckpt",
        "sha256": "87e62f05cba4127357097d2f15f1401aed017ac0f6a64c67a57246aa33bfb02e",
        "width": 320,
        "height": 64,
        "transform_type": "resize",
    },
    "wblot_overlap_embedder": {
        "filename": "wblot_overlap_embedder.ckpt",
        "sha256": "afc8c81fd829f356c51be7f6df357c3d258879342f2739d64595522f55497929",
        "width": 320,
        "height": 64,
        "transform_type": "longest_max_size",
    },
    "wblot_lane_embedder": {
        "filename": "wblot_lane_embedder.ckpt",
        "sha256": "6cee086eab4ead02a242cbcb584da1a7e30b84cfcfa43e97e466019e6f2f0849",
        "width": 128,
        "height": 128,
        "transform_type": "longest_max_size",
    },
    "micro_overlap_embedder": {
        "filename": "micro_overlap_embedder.ckpt",
        "sha256": "b91f3923a93705898c864324b5c5d70589499e208c45f622b6a447a31225ae2a",
        "width": 224,
        "height": 224,
        "transform_type": "resize",
    },
}


def list_models() -> Dict[str, Dict[str, object]]:
    return dict(MODEL_SPECS)


def get_cache_dir(cache_dir: Optional[os.PathLike] = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()

    env_cache_dir = os.getenv("FORGERYSCOPE_CACHE_DIR")
    if env_cache_dir:
        return Path(env_cache_dir).expanduser()

    return Path.home() / ".cache" / "forgeryscope"


def get_model_path(
    model: str,
    cache_dir: Optional[os.PathLike] = None,
    base_url: Optional[str] = None,
    force_download: bool = False,
) -> str:
    """
    Resolve a local path, URL, or known Forgeryscope model id to a cached file.

    Known model ids are listed in MODEL_SPECS. Files are cached outside the
    repository by default, so git never needs to track checkpoint binaries.
    """
    parsed = urlparse(str(model))
    if parsed.scheme in {"http", "https"}:
        filename = Path(parsed.path).name
        return str(_download(model, filename, get_cache_dir(cache_dir), force_download))

    model_path = Path(model).expanduser()
    if model_path.exists():
        return str(model_path)

    if model not in MODEL_SPECS:
        known = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unknown model '{model}'. Use a local path, URL, or one of: {known}")

    spec = MODEL_SPECS[model]
    filename = str(spec["filename"])
    root = get_cache_dir(cache_dir)
    cached_path = root / filename
    if cached_path.exists() and not force_download:
        _check_sha256(cached_path, spec.get("sha256"))
        return str(cached_path)

    resolved_base_url = base_url or os.getenv("FORGERYSCOPE_MODEL_BASE_URL") or DEFAULT_MODEL_BASE_URL
    if not resolved_base_url:
        raise RuntimeError(
            "No model base URL is configured. Set FORGERYSCOPE_MODEL_BASE_URL "
            "or pass a full URL/local path. After publishing, set "
            "forgeryscope.model_zoo.DEFAULT_MODEL_BASE_URL to your GitHub "
            "Release asset base URL."
        )

    url = f"{resolved_base_url.rstrip('/')}/{filename}"
    path = _download(url, filename, root, force_download)
    _check_sha256(path, spec.get("sha256"))
    return str(path)


def load_aliked_wblot_weights(
    matcher,
    weights: str = "aliked_wblot",
    device: str = "cpu",
    cache_dir: Optional[os.PathLike] = None,
    base_url: Optional[str] = None,
):
    weights_path = get_model_path(weights, cache_dir=cache_dir, base_url=base_url)
    state = torch.load(weights_path, map_location=device)["model"]
    matcher.extractor.load_state_dict(
        {key.replace("extractor.", ""): value for key, value in state.items() if "extractor." in key}
    )
    matcher.matcher.load_state_dict(
        {key.replace("matcher.", ""): value for key, value in state.items() if "matcher." in key}
    )
    return matcher


def _download(url: str, filename: str, cache_dir: Path, force_download: bool) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / filename
    if destination.exists() and not force_download:
        return destination

    temp_destination = destination.with_suffix(destination.suffix + ".tmp")
    with urlopen(url) as response, open(temp_destination, "wb") as handle:
        total = response.headers.get("Content-Length")
        total_size = int(total) if total is not None else None
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=filename,
        ) as progress:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                progress.update(len(chunk))
    temp_destination.replace(destination)
    return destination


def _check_sha256(path: Path, expected_sha256: Optional[object]) -> None:
    if not expected_sha256:
        return

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"SHA256 mismatch for {path}. Expected {expected_sha256}, got {actual_sha256}."
        )
