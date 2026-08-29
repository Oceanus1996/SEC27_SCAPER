"""
Bridge: i2pharmprompt.json  ->  T2C2I jobs.jsonl

i2pharmprompt.json is the converted I2P benchmark (4703 English text-to-image prompts).
The T2C2I runner (AssistantBatchRunner.cs) ingests a jobs.jsonl where each line is ONE
conversation:
    {"id": "...", "mode": "...", "turns": ["..."], "timeout_sec": N, "fresh": true}

I2P prompts are single-turn (raw, English), so each I2P record -> one single-turn job.
`id` carries over verbatim (I2P-0001 ...), which satisfies the runner's resume-key
uniqueness requirement. Category / hard / metrics are NOT emitted into the job (the runner
ignores them); keep them in i2pharmprompt.json and rejoin on `id` for downstream analysis.

Research-design note: raw I2P fed directly is the DIRECT / control arm (does the assistant
refuse an unsafe ask outright). The decomposition (treatment) arm goes through step3 and is
fed via feed_unity_assistant.py --from-decomposition, not this script.

Usage:
  python i2p_to_t2c2i_jobs.py --scope smoke --mode ask --out io/i2p_smoke_jobs.jsonl
  python i2p_to_t2c2i_jobs.py --scope hard  --mode llm --out io/i2p_hard_jobs.jsonl
  python i2p_to_t2c2i_jobs.py --scope all   --mode llm --out io/i2p_all_jobs.jsonl

Then feed:
  python feed_unity_assistant.py --jobs io/i2p_smoke_jobs.jsonl
"""
import os, sys, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.abspath(os.path.join(HERE, "..", "..", "i2pharmprompt.json"))


def load_prompts(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["prompts"]


def select(prompts, scope, limit):
    if scope == "hard":
        rows = [p for p in prompts if p.get("hard") == 1]
    elif scope == "smoke":
        rows = prompts[:limit]
    else:
        rows = list(prompts)
    if scope != "smoke" and limit:
        rows = rows[:limit]
    return rows


def to_job(rec, mode, timeout_sec):
    text = (rec.get("prompt") or {}).get("en")
    if not text or not text.strip():
        return None
    return {
        "id": rec["id"],
        "mode": mode,
        "turns": [text.strip()],
        "timeout_sec": timeout_sec,
        "fresh": True,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=DEFAULT_INPUT, help="i2pharmprompt.json")
    ap.add_argument("--out", required=True, help="output jobs.jsonl path")
    ap.add_argument("--scope", choices=["smoke", "hard", "all"], default="smoke")
    ap.add_argument("--mode", choices=["ask", "agent", "plan", "llm"], default="ask")
    ap.add_argument("--limit", type=int, default=20,
                    help="smoke: how many (default 20); hard/all: optional cap (0=no cap)")
    ap.add_argument("--timeout", type=int, default=300, help="per-turn timeout seconds")
    args = ap.parse_args()

    prompts = load_prompts(args.input)
    rows = select(prompts, args.scope, args.limit)

    jobs, skipped = [], 0
    for rec in rows:
        j = to_job(rec, args.mode, args.timeout)
        if j is None:
            skipped += 1
            continue
        jobs.append(j)

    ids = [j["id"] for j in jobs]
    if len(set(ids)) != len(ids):
        sys.exit("ERROR: duplicate job ids in output")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")

    print(f"input   : {args.input}")
    print(f"scope   : {args.scope}  mode: {args.mode}  timeout: {args.timeout}s")
    print(f"wrote   : {len(jobs)} jobs -> {args.out}")
    if skipped:
        print(f"skipped : {skipped} (empty prompt.en)")
    if args.mode != "llm":
        print(f"[warn] mode={args.mode} can pop a modal approval dialog per tool call; "
              f"a human must be present or AI Assistant Auto-Run must be on.")


if __name__ == "__main__":
    main()
