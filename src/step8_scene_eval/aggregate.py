"""Pure logic for step8: discover screenshots, join to original I2P prompts,
aggregate per-view scores into per-scene rows. No torch/model dependency here so
it stays unit-testable with a stub evaluator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

HEADS = ("sexual", "violent", "disturbing", "hateful", "political")
PRIMARY_HEADS = HEADS[:4]
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class SceneImages:
    scene_id: str
    paths: list[Path]


def discover_scenes(root: Path) -> list[SceneImages]:
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"screenshots root does not exist: {root}")
    scenes: list[SceneImages] = []
    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sorted(
            p
            for p in scene_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if images:
            scenes.append(SceneImages(scene_dir.name, images))
    if not scenes:
        raise ValueError(f"no scene folders with images under: {root}")
    return scenes


def load_i2p_index(path: Path) -> dict[str, dict]:
    """id -> {prompt_en, category, hard, metrics} from i2pharmprompt.json."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    index: dict[str, dict] = {}
    for rec in data["prompts"]:
        index[rec["id"]] = {
            "prompt_en": ((rec.get("prompt") or {}).get("en") or "").strip(),
            "category": (rec.get("category") or {}).get("en"),
            "hard": rec.get("hard"),
            "orig_metrics": rec.get("metrics") or {},
        }
    return index


def build_view_records(
    scene_id: str,
    paths: Sequence[Path],
    scored: Sequence[dict],
    threshold: float,
) -> list[dict]:
    if len(paths) != len(scored):
        raise ValueError(
            f"{scene_id}: scorer returned {len(scored)} rows for {len(paths)} images"
        )
    records = []
    for path, row in zip(paths, scored):
        scores = {h: float(row["scores"][h]) for h in HEADS}
        unsafe = {h: scores[h] > threshold for h in HEADS}
        records.append(
            {
                "scene_id": scene_id,
                "view_id": Path(path).stem,
                "image_path": Path(path).name,
                "alignment": float(row["alignment"]),
                "scores": scores,
                "unsafe": unsafe,
                "primary_unsafe": any(unsafe[h] for h in PRIMARY_HEADS),
            }
        )
    return records


def aggregate_scene(
    scene_id: str,
    view_records: Sequence[dict],
    meta: dict | None,
    threshold: float,
) -> dict:
    n = len(view_records)
    aligns = [v["alignment"] for v in view_records]
    row = {
        "scene_id": scene_id,
        "matched_i2p": meta is not None,
        "view_count": n,
        "alignment_max": max(aligns),
        "alignment_mean": sum(aligns) / n,
        "harm_max": {h: max(v["scores"][h] for v in view_records) for h in HEADS},
        "unsafe_fraction": {
            h: sum(v["unsafe"][h] for v in view_records) / n for h in HEADS
        },
        "any_view_unsafe": any(v["primary_unsafe"] for v in view_records),
        "threshold": threshold,
    }
    if meta is not None:
        row["category"] = meta.get("category")
        row["hard"] = meta.get("hard")
        row["prompt_en"] = meta.get("prompt_en")
        row["orig_metrics"] = meta.get("orig_metrics")
    return row


def scene_csv_rows(scene_records: Sequence[dict]) -> tuple[list[str], list[list]]:
    header = (
        ["scene_id", "matched_i2p", "view_count", "alignment_max", "alignment_mean"]
        + [f"harm_max_{h}" for h in HEADS]
        + [f"unsafe_frac_{h}" for h in HEADS]
        + ["any_view_unsafe", "hard", "category", "orig_q16_pct",
           "orig_nudity_pct", "orig_inappropriate_pct", "orig_prompt_toxicity"]
    )
    rows = []
    for r in scene_records:
        om = r.get("orig_metrics") or {}
        cat = r.get("category")
        rows.append(
            [
                r["scene_id"], r["matched_i2p"], r["view_count"],
                round(r["alignment_max"], 5), round(r["alignment_mean"], 5),
                *[round(r["harm_max"][h], 5) for h in HEADS],
                *[round(r["unsafe_fraction"][h], 5) for h in HEADS],
                r["any_view_unsafe"], r.get("hard"),
                "|".join(cat) if isinstance(cat, list) else (cat or ""),
                om.get("q16_percentage"), om.get("nudity_percentage"),
                om.get("inappropriate_percentage"), om.get("prompt_toxicity"),
            ]
        )
    return header, rows
