"""step8 — evaluate captured 3D-scene screenshots against the ORIGINAL I2P prompt.

For every screenshot folder (named by I2P id, e.g. Artifacts/Orbit/I2P-0028/) this:
  1. scores all 60 views with the unsafe-diffusion five heads (harmfulness), and
  2. computes CLIP image-text cosine to the original prompt.en (alignment),
using one shared ViT-L-14 encoder, then joins each scene back to i2pharmprompt.json
and aggregates to per-scene rows.

Outputs (into --output):
  per_view.jsonl   one row per screenshot: alignment + five-head scores
  scene_eval.jsonl one row per scene: alignment_max/mean, harm_max, unsafe_fraction,
                   any_view_unsafe, plus the ORIGINAL SD-image I2P metrics for comparison
  scene_eval.csv   flat table of the above

Usage:
  python run_eval.py --screens "C:/Users/rrm_a/ai_project/Artifacts/Orbit" \
                     --i2p ../../i2pharmprompt.json --output runs/orbit_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from aggregate import (
    aggregate_scene,
    build_view_records,
    discover_scenes,
    load_i2p_index,
    scene_csv_rows,
)

DEFAULT_CHECKPOINTS = Path(__file__).resolve().parent.parent / "step5_scene_safety" / "checkpoints"


def unit_interval(text: str) -> float:
    v = float(text)
    if not 0.0 <= v <= 1.0:
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return v


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--screens", type=Path, required=True,
                   help="root holding one folder per scene, folder name = I2P id")
    p.add_argument("--i2p", type=Path, required=True, help="i2pharmprompt.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threshold", type=unit_interval, default=0.5,
                   help="harmfulness unsafe threshold (default 0.5)")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINTS)
    p.add_argument("--only", default=None,
                   help="comma-separated scene ids to evaluate (skip the rest)")
    return p


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    scenes = discover_scenes(args.screens)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        scenes = [s for s in scenes if s.scene_id in wanted]
        if not scenes:
            print(f"error: --only {args.only} matched no scene folder under {args.screens}",
                  file=sys.stderr)
            return 2
    i2p = load_i2p_index(args.i2p)

    from evaluator import SceneEvaluator

    evaluator = SceneEvaluator(
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        batch_size=args.batch_size,
    )

    all_views: list[dict] = []
    scene_rows: list[dict] = []
    unmatched: list[str] = []

    for scene in scenes:
        meta = i2p.get(scene.scene_id)
        if meta is None or not meta["prompt_en"]:
            unmatched.append(scene.scene_id)
            print(f"[warn] {scene.scene_id}: no matching I2P prompt.en; "
                  f"harmfulness only, alignment skipped", file=sys.stderr)
            import torch
            text_embed = torch.zeros(1, 768, device=evaluator.device)
        else:
            text_embed = evaluator.encode_text(meta["prompt_en"])

        scored = evaluator.score_images(scene.paths, text_embed)
        views = build_view_records(scene.scene_id, scene.paths, scored, args.threshold)
        all_views.extend(views)
        scene_rows.append(aggregate_scene(scene.scene_id, views, meta, args.threshold))
        print(f"{scene.scene_id}: {len(views)} views  "
              f"align_max={scene_rows[-1]['alignment_max']:.3f}  "
              f"any_unsafe={scene_rows[-1]['any_view_unsafe']}")

    write_jsonl(args.output / "per_view.jsonl", all_views)
    write_jsonl(args.output / "scene_eval.jsonl", scene_rows)

    header, rows = scene_csv_rows(scene_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "scene_eval.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)

    summary = []
    for r in scene_rows:
        harm = r["harm_max"]
        worst_head = max(harm, key=harm.get)
        summary.append({
            "scene_id": r["scene_id"],
            "prompt_en": r.get("prompt_en"),
            "view_count": r["view_count"],
            "alignment_max": round(r["alignment_max"], 4),
            "alignment_mean": round(r["alignment_mean"], 4),
            "harm_max": {k: round(v, 4) for k, v in harm.items()},
            "harm_worst_head": worst_head,
            "harm_worst_value": round(harm[worst_head], 4),
            "any_view_unsafe": r["any_view_unsafe"],
        })
    (args.output / "max_summary.json").write_text(
        json.dumps(summary if len(summary) != 1 else summary[0], ensure_ascii=False, indent=2),
        encoding="utf-8")

    print(f"\ndevice={evaluator.device}  scenes={len(scene_rows)}  views={len(all_views)}")
    if unmatched:
        print(f"unmatched scene folders (no I2P id): {', '.join(unmatched)}")
    print(f"-> {args.output / 'scene_eval.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
