"""Shared OpenCLIP evaluator: five-head harmfulness + image/text alignment.

One ViT-L-14 forward pass per image yields BOTH:
  * unsafe-diffusion five-head scores (reusing step5's heads + checkpoints), and
  * a normalized image embedding for cosine similarity against the prompt text.

Reusing the exact same visual backbone as step5 keeps harmfulness and alignment
on one consistent encoder and avoids loading the ~1.7GB ViT-L twice.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError

_STEP5 = Path(__file__).resolve().parent.parent / "step5_scene_safety"
if str(_STEP5) not in sys.path:
    sys.path.insert(0, str(_STEP5))

from model import (
    ProjectionHead,
    _create_open_clip,
    _load_state_dict,
    ensure_checkpoints,
    resolve_device,
)
from scene_safety import HEADS

MODEL_NAME = "ViT-L-14-quickgelu"


class SceneEvaluator:
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

        import open_clip

        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)

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

    @torch.inference_mode()
    def encode_text(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer([text]).to(self.device)
        embed = self.encoder.encode_text(tokens).to(dtype=torch.float32)
        return F.normalize(embed, dim=-1)

    @torch.inference_mode()
    def score_images(
        self, paths: Sequence[Path], text_embed: torch.Tensor
    ) -> list[dict]:
        """For each image: five-head scores + cosine alignment to `text_embed`."""
        results: list[dict] = []
        for start in range(0, len(paths), self.batch_size):
            batch = paths[start : start + self.batch_size]
            images = torch.stack([self._preprocess(Path(p)) for p in batch]).to(
                self.device
            )
            features = self.encoder.encode_image(images).to(dtype=torch.float32)

            heads = {
                name: head(features).squeeze(1).detach().cpu().tolist()
                for name, head in self.heads.items()
            }
            img_embed = F.normalize(features, dim=-1)
            cos = (img_embed @ text_embed.T).squeeze(1).detach().cpu().tolist()

            for i in range(len(batch)):
                results.append(
                    {
                        "scores": {name: float(heads[name][i]) for name in HEADS},
                        "alignment": float(cos[i]),
                    }
                )
        return results
