# SCAPER — Artifact

This artifact accompanies the paper *SCAPER: Compositional Jailbreaks against Unity's Text-to-3D Generation*. This anonymous release supports reproduction of our safety evaluation of Unity's integrated text-to-3D generation pipeline.

> **⚠️ Ethics / Responsible Use**
>
> This artifact is intended solely for security research and defensive evaluation. We **do not release harmful 3D assets, the complete adversarial prompt set, or directly runnable attack code**. The seed prompts are drawn from the public I2P benchmark. See `ETHICS.md` for details.

---

## 1. Contents

```text
src/decompose/              Textual Scene Lifting + Scene-Level Decomposition
  step1.py … step4.py       four-stage decomposition prompt templates
  step1_5.py step1_7.py     intermediate variants
  step_all.py               single-pass variant
  common.py                 dataset / prompt loading and generation driver
  measure.py                measurement-only batch pipeline (3 analysis steps)
  qwen_base.py              direct Qwen3.5-9B-Base calls for prompt testing

src/step5_scene_safety/     scene-level safety scoring
  model.py                  five-head classifier wrapper
  scene_safety.py           per-scene aggregation

src/step8_scene_eval/       60-view rendering evaluation
  evaluator.py              per-view scoring
  run_eval.py               evaluation entry point
  aggregate.py              per-view → per-scene aggregation

src/unity_comm/             Unity job construction and input feeding
  i2p_to_t2c2i_jobs.py      I2P seeds → text-to-3D job specifications
  build_jobs.py             job-file construction
  feed_unity_assistant.py   drives the Unity Assistant
  export_record.py          exports annotations to spreadsheets

reproduce/seed_selection.py
                            reproduces the selection of 286 seeds from I2P

configs/viewpoints_60.csv
                            60-view capture protocol (Table 5 in the paper)

data/
  rq1_data.xlsx             per-prompt bypass / harm results across three Unity tiers
                            → supports Table 2 (RQ1) and Table 4 (RQ3)

  ss.xlsx, ss.csv           semantic-similarity values — see §4; not usable as-is

  annotations/
    master_annotation_table.csv
                            earlier CSV version of the annotation table

    stage_validation_50.csv
                            three-reviewer validation of the 50-prompt subset

tables_tab2_filled.tex      LaTeX source for Table 2 with per-category N filled in

requirements.txt            evaluation-stage dependencies