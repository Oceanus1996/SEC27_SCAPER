"""Measurement pipeline (batched, single GPU).

Three steps, all analysis/measurement: no generation, no sanitization, not designed to pass review:
    step1 attribution   {scene_id,category,prompt} -> entities/relations/environment + harm attribution (base model)
    step2 decomposition step1 JSON                  -> prefab/relation/factor/step (base model)
    step3 normalization step2 JSON                  -> canonical_relations + reference validation (base model)

Output: one jsonl per step plus the step3 reference-validation statistics.

Change the variables in main(), then run:
    /home/ruomai/.venv/bin/python measure.py

The prompts live in prompts/measure.py (STEP1 / STEP2 / STEP3) and can be edited.
"""

import json
import re
from collections import Counter
from pathlib import Path

import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          StoppingCriteria, StoppingCriteriaList)

import common

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
_LOADED = {}


def load_measure_prompt(name):
    """Load STEP1 / STEP2 / STEP3 from prompts/measure.py. Re-read every time, so edits take effect on the next run."""
    ns = {}
    exec(compile((PROMPT_DIR / "measure.py").read_text(encoding="utf-8"),
                 "measure.py", "exec"), ns)
    print("prompts",ns[name])
    return ns[name]


def get_model(model_path, device):
    key = (model_path, device)
    if key not in _LOADED:
        print(f"[+] loading {model_path} -> {device}")
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True).to(device)
        model.eval()
        _LOADED[key] = (tok, model)
        print("[+] ready")
    return _LOADED[key]


class StopOnBalancedJSON(StoppingCriteria):
    """Stop this sequence as soon as the top-level JSON object is balanced.

    A base model has no notion of EOS: after finishing once it starts over, until max_new_tokens is exhausted.
    parse_json only takes the first balanced object anyway, so the rest is waste and would make hit_limit permanently True.
    """

    def __init__(self, tok, in_len, n_rows, device, prefilled):
        self.tok, self.in_len = tok, in_len
        self.state = [[1 if prefilled else 0, bool(prefilled), False, False]
                      for _ in range(n_rows)]
        self.seen = [0] * n_rows
        self.done = torch.zeros(n_rows, dtype=torch.bool, device=device)

    def __call__(self, input_ids, scores, **kw):
        for i in range(input_ids.shape[0]):
            if self.done[i]:
                continue
            new = input_ids[i, self.in_len + self.seen[i]:]
            if new.numel() == 0:
                continue
            self.seen[i] += int(new.numel())
            depth, started, in_str, esc = self.state[i]
            for ch in self.tok.decode(new, skip_special_tokens=True):
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth, started = depth + 1, True
                elif ch == "}":
                    depth -= 1
                    if started and depth <= 0:
                        self.done[i] = True
                        break
            self.state[i] = [depth, started, in_str, esc]
        return self.done


def generate_batch(model_path, prompts, device="cuda:0", batch_size=8,
                   max_new_tokens=3000, json_prefill=True, chat=None,
                   required=()):
    """Feed several records at once. Returns a list of (raw, parsed, status, n_tokens) the same length as prompts.

    required: keys that must exist at the top level; missing ones count as incomplete (not carried to the next step).
    """
    tok, model = get_model(model_path, device)

    if chat is None:
        chat = "Base" not in Path(model_path).name

    out_all = []
    for s in range(0, len(prompts), batch_size):
        chunk = prompts[s:s + batch_size]

        texts = []
        for p in chunk:
            t = p
            if chat:
                t = tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True)
            if json_prefill:
                t = t.rstrip() + "\n{"
            texts.append(t)

        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=not chat).to(model.device)
        in_len = enc["input_ids"].shape[1]

        stopper = StopOnBalancedJSON(tok, in_len, len(chunk), model.device,
                                     json_prefill)
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                stopping_criteria=StoppingCriteriaList([stopper]),
            )

        for row in out:
            gen = row[in_len:]
            raw = strip_think(tok.decode(gen, skip_special_tokens=True))
            n = int((gen != tok.pad_token_id).sum())
            if json_prefill and not raw.lstrip().startswith("{"):
                raw = "{" + raw
            parsed = parse_json(raw)
            status = classify(raw, parsed, n >= max_new_tokens, required)
            out_all.append((raw, parsed, status, n))

        print(f"    batch {s // batch_size + 1}: {len(chunk)} records done")

    return out_all


def strip_think(text):
    """Drop the chain-of-thought and keep only the body. The saved raw_output uses this cleaned version."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"<think>.*", "", text, flags=re.S)
    return text.split("</think>", 1)[-1].strip()


def parse_json(raw):
    body = re.sub(r"```(?:json)?", "", raw)
    depth = start = 0
    in_str = esc = False
    for i, ch in enumerate(body):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if not depth:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if not depth:
                blob = body[start:i + 1]
                fixed = re.sub(r",(\s*[}\]])", r"\1", blob)
                smart = fixed.replace("“", '"').replace("”", '"')
                for cand in (blob, fixed, smart):
                    try:
                        obj = json.loads(cand)
                    except ValueError:
                        continue
                    if isinstance(obj, dict):
                        return obj
    return None


def classify(raw, parsed, hit_limit, required=()):
    if parsed is not None:
        missing = [k for k in required if k not in parsed]
        return f"incomplete:{','.join(missing)}" if missing else "ok"
    if re.search(r"我无法|我不能|我拒绝|抱歉|无法生成|无法提供"
                 r"|I can(?:no|')t |I cannot |I'm sorry|unable to assist", raw):
        return "refused"
    return "truncated" if hit_limit else "parse_failed"


def dump(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[+] wrote {len(records)} records -> {path}")


REQUIRED = {
    "STEP1": ("scene_id", "entities", "key_relational_factors", "environment",
              "harm_analysis", "harm_relations"),
    "STEP2": ("scene_id", "prefabs", "relations", "steps"),
    "STEP3": ("scene_id", "prefabs", "canonical_relations", "steps", "validation"),
    "STEP4": (),
}


def keep_ok(step_name, records):
    """Filter the records that can proceed and print why the rest were dropped, instead of silently passing an empty list."""
    ok = [r for r in records if r["status"] == "ok"]
    print(f"[+] {step_name}: {len(ok)}/{len(records)} ok  "
          f"{dict(Counter(r['status'] for r in records))}")
    if records and not ok:
        print(f"[!] {step_name} has no usable record; the following steps will all be empty. "
              f"Inspect raw_output for the shape before re-running")
    return ok


def run(input_path, ids=None, limit=None,
        base_model="/mnt/hdd1/ruomai/models/Qwen3.5-9B-Base",
        device="cuda:0", batch_size=8, out_dir="/home/ruomai/data/steps"):

    rows = common.read_data(input_path)
    if ids:
        rows = [r for r in rows if r["id"] in ids]
    if limit:
        rows = rows[:limit]
    print(f"[+] {len(rows)} records entering the measurement pipeline")

    print("\n===== STEP1  =====")
    p1 = load_measure_prompt("STEP1")
    prompts = [p1.replace("{input_json}", json.dumps(
        {"scene_id": str(r["id"]), "category": r.get("category") or "",
         "prompt": r["source_prompt"]}, ensure_ascii=False, indent=2))
        for r in rows]
    res1 = generate_batch(base_model, prompts, device, batch_size,
                          max_new_tokens=3000, required=REQUIRED["STEP1"])

    step1 = []
    for r, (raw, parsed, status, n) in zip(rows, res1):
        step1.append({"id": r["id"], "category": r.get("category"),
                      "source_prompt": r["source_prompt"], "status": status,
                      "n_tokens": n, "parsed_output": parsed, "raw_output": raw})
    dump(f"{out_dir}/measure_step1.jsonl", step1)

    ok1 = keep_ok("step1", step1)

    print("\n===== STEP2  =====")
    p2 = load_measure_prompt("STEP2")
    prompts = [p2.replace("{step1_json}", json.dumps(s["parsed_output"], ensure_ascii=False, indent=2))
               for s in ok1]
    res2 = generate_batch(base_model, prompts, device, batch_size,
                          max_new_tokens=3000, required=REQUIRED["STEP2"])

    step2 = []
    for s, (raw, parsed, status, n) in zip(ok1, res2):
        step2.append({"id": s["id"], "category": s.get("category"),
                      "status": status, "n_tokens": n,
                      "parsed_output": parsed, "raw_output": raw})
    dump(f"{out_dir}/measure_step2.jsonl", step2)

    ok2 = keep_ok("step2", step2)

    print("\n===== STEP3  =====")
    p3 = load_measure_prompt("STEP3")
    prompts = [p3.replace("{step2_json}", json.dumps(s["parsed_output"], ensure_ascii=False, indent=2))
               for s in ok2]
    res3 = generate_batch(base_model, prompts, device, batch_size,
                          max_new_tokens=3000, required=REQUIRED["STEP3"])

    step3 = []
    for s, (raw, parsed, status, n) in zip(ok2, res3):
        step3.append({"id": s["id"], "category": s.get("category"),
                      "status": status, "n_tokens": n,
                      "parsed_output": parsed, "raw_output": raw})
    dump(f"{out_dir}/measure_step3.jsonl", step3)

    ok3 = keep_ok("step3", step3)

    valid = [s for s in ok3
             if all((s["parsed_output"].get("validation") or {}).get(k) is True
                    for k in ("all_prefab_references_valid",
                              "all_relation_references_valid",
                              "all_factor_references_valid"))]
    print(f"[+] step3 references fully consistent: {len(valid)}/{len(ok3)}")


def read_jsonl(path):
    """Read the previous step's output line by line, skipping blanks. read_data only accepts a whole JSON file and is not used here."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_2(step3_path="/home/ruomai/data/steps/measure_step3.jsonl",
          ids=None, limit=None,
          base_model="/mnt/hdd1/ruomai/models/Qwen3.5-9B-Base",
          device="cuda:0", batch_size=8,
          out_dir="/home/ruomai/data/steps"):
    """Run STEP4 on its own: read the step3 jsonl -> STEP4 prompt -> measure_step4.jsonl.

    Only rows with status == ok and a parsed_output are used; rows where step3 failed have no JSON to feed.
    """
    rows = read_jsonl(step3_path)
    ok3 = [r for r in rows if r.get("status") == "ok" and r.get("parsed_output")]
    print(f"[+] {step3_path}: {len(ok3)}/{len(rows)} usable  "
          f"{dict(Counter(r.get('status') for r in rows))}")

    if ids:
        ids = set(ids)
        ok3 = [r for r in ok3 if r["id"] in ids]
    if limit:
        ok3 = ok3[:limit]
    if not ok3:
        print("[!] no usable input, skipping STEP4. Check status in measure_step3.jsonl first")
        return []
    print(f"[+] {len(ok3)} records entering STEP4")

    print("\n===== STEP4  =====")
    p4 = load_measure_prompt("STEP4")
    if not p4 or "{step3_json}" not in p4:
        raise ValueError("STEP4 in prompts/measure.py is still empty, or has no "
                         "{step3_json} placeholder. Finish the prompt before running.")

    prompts = [p4.replace("{step3_json}", json.dumps(
        r["parsed_output"], ensure_ascii=False, indent=2)) for r in ok3]
    res4 = generate_batch(base_model, prompts, device, batch_size,
                          max_new_tokens=3000, required=REQUIRED["STEP4"])

    step4 = [{"id": r["id"], "category": r.get("category"),
              "status": status, "n_tokens": n,
              "parsed_output": parsed, "raw_output": raw}
             for r, (raw, parsed, status, n) in zip(ok3, res4)]
    dump(f"{out_dir}/measure_step4.jsonl", step4)

    keep_ok("step4", step4)
    return step4


def main():

    run_2(
        step3_path="/home/ruomai/data/steps/measure_step3.jsonl",
        ids=None,
        limit=None,
        device="cuda:1",
        batch_size=8,
    )


if __name__ == "__main__":
    main()
