"""Pure input/output and aggregation logic for scene safety scoring."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


HEADS = ("sexual", "violent", "disturbing", "hateful", "political")
PRIMARY_HEADS = HEADS[:4]
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class ImageItem:
    scene_id: str
    view_id: str
    path: Path
    relative_path: str


class Scorer(Protocol):
    def score(self, paths: Sequence[Path]) -> list[dict[str, float]]:
        """Return one five-head score mapping for every input path."""


def discover_images(root: Path) -> list[ImageItem]:
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"input directory does not exist: {root}")

    items = []
    for scene_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        images = sorted(
            path
            for path in scene_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        for path in images:
            items.append(
                ImageItem(
                    scene_id=scene_dir.name,
                    view_id=path.stem,
                    path=path,
                    relative_path=path.relative_to(root).as_posix(),
                )
            )

    if not items:
        raise ValueError(f"no supported images found under: {root}")
    keys = [(item.scene_id, item.view_id) for item in items]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate view IDs found within a scene")
    return items


def _validate_threshold(threshold: float) -> float:
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be a finite value in [0, 1]")
    return threshold


def validate_scores(scores: dict[str, float]) -> dict[str, float]:
    if not isinstance(scores, dict) or set(scores) != set(HEADS):
        raise ValueError("scores must contain exactly the five named heads")
    try:
        clean = {head: float(scores[head]) for head in HEADS}
    except (TypeError, ValueError) as exc:
        raise ValueError("scores must be numeric values in [0, 1]") from exc
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in clean.values()
    ):
        raise ValueError("scores must be finite values in [0, 1]")
    return clean


def build_image_records(
    items: Sequence[ImageItem],
    score_rows: Sequence[dict[str, float]],
    threshold: float,
) -> list[dict]:
    threshold = _validate_threshold(threshold)
    if len(items) != len(score_rows):
        raise ValueError(
            f"scorer returned {len(score_rows)} rows for {len(items)} images"
        )

    records = []
    for item, raw_scores in zip(items, score_rows):
        scores = validate_scores(raw_scores)
        unsafe = {head: scores[head] > threshold for head in HEADS}
        records.append(
            {
                "scene_id": item.scene_id,
                "view_id": item.view_id,
                "image_path": item.relative_path,
                "scores": scores,
                "unsafe": unsafe,
                "primary_unsafe": any(unsafe[head] for head in PRIMARY_HEADS),
            }
        )
    return records


def aggregate_scene_records(image_records: Sequence[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for record in image_records:
        grouped.setdefault(record["scene_id"], []).append(record)

    scene_records = []
    for scene_id in sorted(grouped):
        views = grouped[scene_id]
        scene_records.append(
            {
                "scene_id": scene_id,
                "view_count": len(views),
                "max_scores": {
                    head: max(view["scores"][head] for view in views)
                    for head in HEADS
                },
                "unsafe_fraction": {
                    head: sum(view["unsafe"][head] for view in views) / len(views)
                    for head in HEADS
                },
                "any_view_unsafe": any(view["primary_unsafe"] for view in views),
            }
        )
    return scene_records


def write_jsonl_atomic(path: Path, records: Sequence[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                line = json.dumps(
                    record, ensure_ascii=False, separators=(",", ":")
                )
                handle.write(line + "\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run_pipeline(
    input_root: Path,
    output_dir: Path,
    scorer: Scorer,
    threshold: float = 0.5,
) -> tuple[list[dict], list[dict]]:
    items = discover_images(Path(input_root))
    score_rows = scorer.score([item.path for item in items])
    image_records = build_image_records(items, score_rows, threshold)
    scene_records = aggregate_scene_records(image_records)

    output_dir = Path(output_dir)
    write_jsonl_atomic(output_dir / "per_image.jsonl", image_records)
    write_jsonl_atomic(output_dir / "per_scene.jsonl", scene_records)
    return image_records, scene_records
