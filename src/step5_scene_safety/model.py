"""PyTorch/OpenCLIP adapter for the official unsafe-diffusion heads."""

import inspect
import shutil
import urllib.request
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn

from scene_safety import HEADS


UPSTREAM_COMMIT = "9ad261669b9dbdbf1a6ece8ff48b2cb6fdfec586"
CHECKPOINT_BASE_URL = (
    "https://raw.githubusercontent.com/YitingQu/unsafe-diffusion/"
    f"{UPSTREAM_COMMIT}/checkpoints/multi-headed"
)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available; use --device cpu or auto")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return torch.device(requested)


def _download(url: str, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "scene-safety-pipeline"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def ensure_checkpoints(directory: Path) -> dict[str, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for head in HEADS:
        path = directory / f"{head}.pt"
        if not path.is_file():
            _download(f"{CHECKPOINT_BASE_URL}/{head}.pt", path)
        paths[head] = path
    return paths


class ProjectionHead(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(384),
            nn.Linear(384, 1),
            nn.Sigmoid(),
        )


def _create_open_clip():
    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError(
            "open_clip is not installed; run: pip install -r requirements.txt"
        ) from exc
    return open_clip.create_model_and_transforms(
        "ViT-L-14-quickgelu", pretrained="openai"
    )


def _load_state_dict(path: Path, device: torch.device):
    kwargs = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    return torch.load(path, **kwargs)


class UnsafeDiffusionScorer:
    def __init__(
        self,
        checkpoint_dir: Path,
        device: str = "auto",
        batch_size: int = 16,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch size must be at least 1")
        self.device = resolve_device(device)
        self.batch_size = batch_size

        self.encoder, _, self.preprocess = _create_open_clip()
        self.encoder.to(self.device).eval()
        self.encoder.requires_grad_(False)

        self.heads = {}
        for name, path in ensure_checkpoints(Path(checkpoint_dir)).items():
            head = ProjectionHead().to(self.device)
            head.load_state_dict(_load_state_dict(path, self.device))
            self.heads[name] = head.eval()

    def _preprocess(self, path: Path) -> torch.Tensor:
        try:
            with Image.open(path) as image:
                return self.preprocess(image.convert("RGB"))
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"cannot decode image: {path}") from exc

    def score(self, paths: Sequence[Path]) -> list[dict[str, float]]:
        results = []
        with torch.inference_mode():
            for start in range(0, len(paths), self.batch_size):
                batch_paths = paths[start : start + self.batch_size]
                images = torch.stack(
                    [self._preprocess(Path(path)) for path in batch_paths]
                ).to(self.device)
                features = self.encoder.encode_image(images).to(dtype=torch.float32)
                per_head = {
                    name: head(features).squeeze(1).detach().cpu().tolist()
                    for name, head in self.heads.items()
                }
                for index in range(len(batch_paths)):
                    results.append(
                        {name: float(per_head[name][index]) for name in HEADS}
                    )
        return results
