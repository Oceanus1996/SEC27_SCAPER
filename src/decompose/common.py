import json
import re
from pathlib import Path
from xml.parsers.expat import model

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

_LOADED = {}


def read_data(resource_path):
    """Parse prompts and related information from a JSON dataset."""
    with open(resource_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("records") or data.get("prompts") or data

    rows = []
    for it in items:
        p = it.get("prompt")
        text = p if isinstance(p, str) else (p.get("en") or p.get("zh"))
        if not text:
            continue

        rt = it.get("risk_type")
        rows.append({
            **it,
            "source_prompt": text,
            "category": str(it["id"]).split("-")[0],
            "risk_type": rt.get("code") if isinstance(rt, dict) else rt,
        })

    return rows


def select_prompts(step, prompt_id=None):
    """Select the prompt named by the given step file from the prompts directory.

    For example, prompt_id="V3" in step1.py returns the variable V3.
    """
    path = PROMPT_DIR / f"step{step}.py"
    name = prompt_id or "PROMPT"

    ns = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)

    print(f"[+] using {name} from prompts/step{step}.py")

    return ns[name]


def run_model(model_path, prompt, input_data=None, max_new_tokens=3000,
              temperature=0.0, device="cuda:1", json_prefill=True,
              repetition_penalty=1.0):
    """Run the model and return (raw output, parsed JSON or None, status, token count).

    There are four distinct statuses; do not collapse them into a single error code,
    because they are handled in completely different ways:
      ok            JSON was parsed
      refused       the model refused; more tokens will not help, change model or prompt
      truncated     hit max_new_tokens, just raise the limit
      parse_failed  there is output but it is not JSON, inspect raw_output

    json_prefill   appends a "{" to the end of the prompt to force the model to start
                   writing from inside the JSON. Without it, a base model first
                   improvises (writing <think>, inventing its own constraint list),
                   which wastes thousands of tokens and easily gets stuck in a
                   repetition loop that hits the limit.
    repetition_penalty  defaults to 1.0 (off). Only raise it to around 1.05 if you
                   actually hit repetition; do not raise it further, since JSON
                   naturally contains many repeated tokens (such as "": "").
    """
    key = (model_path, device)

    if key not in _LOADED:
        print(f"[+] loading {model_path} -> {device}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True).to(device)
        model.eval()
        _LOADED[key] = (tokenizer, model)
        print("[+] ready")

    tokenizer, model = _LOADED[key]

    chat = "Base" not in Path(model_path).name
    text = prompt
    if chat:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)

    if json_prefill:
        text = text.rstrip() + "\n{"

    enc = tokenizer(text, return_tensors="pt",
                    add_special_tokens=not chat).to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=0.9 if temperature > 0 else None,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen = out[0][enc["input_ids"].shape[1]:]
    raw = tokenizer.decode(gen, skip_special_tokens=True).strip()
    n_tokens = int(gen.shape[0])

    if json_prefill and not raw.lstrip().startswith("{"):
        raw = "{" + raw
    hit_limit = n_tokens >= max_new_tokens

    body = raw.split("</think>", 1)[-1] if "</think>" in raw else raw
    body = re.sub(r"```(?:json)?", "", body)

    parsed = None
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

                chars = []
                in_s = q_esc = False
                for j, c in enumerate(fixed):
                    if not in_s:
                        chars.append(c)
                        if c == '"':
                            in_s = True
                        continue
                    if q_esc:
                        chars.append(c)
                        q_esc = False
                        continue
                    if c == "\\":
                        chars.append(c)
                        q_esc = True
                        continue
                    if c == '"':
                        k = j + 1
                        while k < len(fixed) and fixed[k] in " \t\r\n":
                            k += 1
                        if k < len(fixed) and fixed[k] not in ",:}]":
                            chars.append('\\"')
                        else:
                            chars.append(c)
                            in_s = False
                        continue
                    chars.append(c)
                escaped = "".join(chars)

                smart = escaped.replace("“", '"').replace("”", '"')

                for attempt in (blob, fixed, escaped, smart):
                    try:
                        obj = json.loads(attempt)
                    except ValueError:
                        continue
                    if isinstance(obj, dict):
                        parsed = obj
                        break

                if parsed is not None:
                    break

    if parsed is not None:
        status = "ok"
    elif re.search(r"我无法|我不能|我拒绝|抱歉"
                   r"|无法生成|无法提供|无法协助"
                   r"|I can(?:no|')t |I cannot |I'm sorry|unable to assist", body):
        status = "refused"
    elif hit_limit:
        status = "truncated"
    else:
        status = "parse_failed"

    return raw, parsed, status, n_tokens


def run_step_1(
    input_path="/home/ruomai/i2p_atomicizable_ranked_prompts.json",
    prompt=None,
    output_path="/home/ruomai/data/steps/step1_v2.jsonl",
    model_path="/mnt/hdd1/ruomai/models/Qwen3.5-9B-Base",
    prompt_id="V2",
    device="cuda:1",
    ids=None,
    limit=None,
    max_new_tokens=3000,
    temperature=0.0,
):
    """Run step 1: read the dataset, call the model, and save the output to a jsonl file.

    Change the defaults above, then run:
        /home/ruomai/.venv/bin/python common.py

    input_path   dataset. An alternative: /home/ruomai/i2pexample.json (40 records)
    prompt_id    variable name inside prompts/step1.py. None means use PROMPT
    model_path   aligned version: /mnt/hdd1/ruomai/models/Qwen3.5-9B
    device       defaults to cuda:1. cuda:0 is often held by a notebook kernel, and
                 competing for it causes OOM in which the process is SIGKILLed
                 directly by the kernel, without even a traceback in the log
    ids          ["I2P-1393"], run only the given records
    limit        run only the first N records

    Each record prints four sections: INPUT / PROMPT / OUTPUT / PARSED.
    """
    if prompt is None:
        prompt = select_prompts(1, prompt_id)

    if "{harm_prompt}" not in prompt:
        raise ValueError("prompt has no {harm_prompt} placeholder to fill")

    data = read_data(input_path)

    if ids:
        data = [r for r in data if r["id"] in ids]
    if limit:
        data = data[:limit]

    print(f"[+] {len(data)} records to process  prompt={prompt_id or 'PROMPT'}")
    if not data:
        return output_path

    stats = {}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(data, 1):
            harm_prompt = row["source_prompt"]
            filled_prompt = prompt.replace("{harm_prompt}", harm_prompt)

            raw, parsed, status, n_tokens = run_model(
                model_path, filled_prompt, row,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=device)

            print(f"\n{'=' * 78}")
            print(f"[{i}/{len(data)}] {row['id']}  status={status}  tokens={n_tokens}")

            print(f"\n{'-' * 30} INPUT (dataset original) {'-' * 30}")
            print(harm_prompt)

            print(f"\n{'-' * 26} PROMPT (full model input) {'-' * 26}")
            print(filled_prompt)

            print(f"\n{'-' * 28} OUTPUT (raw model output) {'-' * 28}")
            print(raw)

            print(f"\n{'-' * 30} PARSED (parsed JSON) {'-' * 30}")
            print(json.dumps(parsed, ensure_ascii=False, indent=2) if parsed
                  else f"(no JSON could be parsed, status={status})")

            result = {
                "id": row["id"],
                "category": row.get("category"),
                "risk_type": row.get("risk_type"),
                "variant": row.get("variant"),
                "rank": row.get("rank"),
                "suitability": row.get("suitability"),
                "atomicizability": row.get("atomicizability"),
                "source_prompt": harm_prompt,
                "prompt_id": prompt_id or "PROMPT",
                "model": model_path,
                "status": status,
                "n_tokens": n_tokens,
                "raw_output": raw,
                "parsed_output": parsed,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            stats[status] = stats.get(status, 0) + 1

    print(f"\n{'=' * 78}")
    print("STEP1 done", dict(sorted(stats.items())))
    print("output", output_path)
    return output_path


def run_step_1_5(
    input_path="/home/ruomai/data/steps/step1_v3.jsonl",
    prompt=None,
    output_path="/home/ruomai/data/steps/step1_5_v3.jsonl",
    model_path="/mnt/hdd1/ruomai/models/Qwen3.5-9B-Base",
    prompt_id=None,
    device="cuda:1",
    ids=None,
    limit=None,
    max_new_tokens=3000,
    temperature=0.0,
):
    """Step 1.5 intermediate check: read the step 1 results, run the check written in
    prompts/step1_5.py, and produce output for step 2.

    The mechanism is the same as step 2: only records with status=ok from the previous
    step are processed, the step 1 parsed_output is filled into {step1_json}, the model
    is called, and the result is saved. The prompt body is edited in prompts/step1_5.py.

    In the output records, parsed_output is the result of this step and step1_output
    keeps the original step 1 output, so step 2 can read both this step's result and
    trace back to step 1.
    """
    if prompt is None:
        prompt = select_prompts("1_5", prompt_id)

    if "{step1_json}" not in prompt:
        raise ValueError("prompt has no {step1_json} placeholder to fill")

    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f if l.strip()]
    data = [r for r in data if r.get("status") == "ok"]

    if ids:
        data = [r for r in data if r["id"] in ids]
    if limit:
        data = data[:limit]

    print(f"[+] {len(data)} records to process (status=ok from previous step)  from {input_path}")
    if not data:
        return output_path

    stats = {}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(data, 1):
            step1_output = row["parsed_output"]
            filled_prompt = prompt.replace(
                "{step1_json}", json.dumps(step1_output, ensure_ascii=False, indent=2))

            raw, parsed, status, n_tokens = run_model(
                model_path, filled_prompt, row,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=device)

            print(f"\n{'=' * 78}")
            print(f"[{i}/{len(data)}] {row['id']}  status={status}  tokens={n_tokens}")

            print(f"\n{'-' * 28} INPUT (step 1 analysis result) {'-' * 28}")
            print("original scene:", row.get("source_prompt"))
            print(json.dumps(step1_output, ensure_ascii=False, indent=2))

            print(f"\n{'-' * 26} PROMPT (full model input) {'-' * 26}")
            print(filled_prompt)

            print(f"\n{'-' * 28} OUTPUT (raw model output) {'-' * 28}")
            print(raw)

            print(f"\n{'-' * 30} PARSED (parsed JSON) {'-' * 30}")
            print(json.dumps(parsed, ensure_ascii=False, indent=2) if parsed
                  else f"(no JSON could be parsed, status={status})")

            result = {
                "id": row["id"],
                "category": row.get("category"),
                "risk_type": row.get("risk_type"),
                "variant": row.get("variant"),
                "rank": row.get("rank"),
                "suitability": row.get("suitability"),
                "atomicizability": row.get("atomicizability"),
                "source_prompt": row.get("source_prompt"),
                "prompt_id": prompt_id or "PROMPT",
                "model": model_path,
                "status": status,
                "n_tokens": n_tokens,
                "step1_output": step1_output,
                "raw_output": raw,
                "parsed_output": parsed,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            stats[status] = stats.get(status, 0) + 1

    print(f"\n{'=' * 78}")
    print("STEP1.5 done", dict(sorted(stats.items())))
    print("output", output_path)
    return output_path


def run_step_1_7(
    input_path="/home/ruomai/data/steps/step1_5_v3.jsonl",
    prompt=None,
    output_path="/home/ruomai/data/steps/step1_v2.jsonl",
    model_path="/mnt/hdd1/ruomai/models/Qwen3.5-9B-Base",
    prompt_id=None,
    device="cuda:1",
    ids=None,
    limit=None,
    max_new_tokens=3000,
    temperature=0.0,
):
    """Step 1.7 intermediate check: compute positional relations.

    The mechanism is the same as step 2: only records with status=ok from the previous
    step are processed, the step 1 parsed_output is filled into {step1_json}, the model
    is called, and the result is saved. The prompt body is edited in prompts/step1_7.py.

    In the output records, parsed_output is the result of this step and step1_output
    keeps the original step 1 output, so step 2 can read both this step's result and
    trace back to step 1.
    """
    if prompt is None:
        prompt = select_prompts("1_7", prompt_id)

    if "{step1_json}" not in prompt:
        raise ValueError("prompt has no {step1_json} placeholder to fill")

    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f if l.strip()]
    data = [r for r in data if r.get("status") == "ok"]

    if ids:
        data = [r for r in data if r["id"] in ids]
    if limit:
        data = data[:limit]

    print(f"[+] {len(data)} records to process (status=ok from previous step)  from {input_path}")
    if not data:
        return output_path

    stats = {}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(data, 1):
            step1_output = row["parsed_output"]
            filled_prompt = prompt.replace(
                "{step1_json}", json.dumps(step1_output, ensure_ascii=False, indent=2))

            raw, parsed, status, n_tokens = run_model(
                model_path, filled_prompt, row,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=device)

            print(f"\n{'=' * 78}")
            print(f"[{i}/{len(data)}] {row['id']}  status={status}  tokens={n_tokens}")

            print(f"\n{'-' * 28} INPUT (step 1 analysis result) {'-' * 28}")
            print("original scene:", row.get("source_prompt"))
            print(json.dumps(step1_output, ensure_ascii=False, indent=2))

            print(f"\n{'-' * 26} PROMPT (full model input) {'-' * 26}")
            print(filled_prompt)

            print(f"\n{'-' * 28} OUTPUT (raw model output) {'-' * 28}")
            print(raw)

            print(f"\n{'-' * 30} PARSED (parsed JSON) {'-' * 30}")
            print(json.dumps(parsed, ensure_ascii=False, indent=2) if parsed
                  else f"(no JSON could be parsed, status={status})")

            result = {
                "id": row["id"],
                "category": row.get("category"),
                "risk_type": row.get("risk_type"),
                "variant": row.get("variant"),
                "rank": row.get("rank"),
                "suitability": row.get("suitability"),
                "atomicizability": row.get("atomicizability"),
                "source_prompt": row.get("source_prompt"),
                "prompt_id": prompt_id or "PROMPT",
                "model": model_path,
                "status": status,
                "n_tokens": n_tokens,
                "step1_output": step1_output,
                "raw_output": raw,
                "parsed_output": parsed,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            stats[status] = stats.get(status, 0) + 1

    print(f"\n{'=' * 78}")
    print("STEP1.5 done", dict(sorted(stats.items())))
    print("output", output_path)
    return output_path

def run_step_2(
    input_path="/home/ruomai/data/steps/step1_v2.jsonl",
    prompt=None,
    output_path="/home/ruomai/data/steps/step2_v2.jsonl",
    model_path="/mnt/hdd1/ruomai/models/Qwen3.5-9B-Base",
    prompt_id="V2",
    device="cuda:1",
    ids=None,
    limit=None,
    max_new_tokens=4000,
    temperature=0.0,
):
    """Run step 2: read the step 1 results, perform relational and causal analysis,
    and save to a jsonl file.

    input_path is the step 1 output. Only status=ok records are processed; failures
    are not carried forward.
    Each record prints four sections: INPUT / PROMPT / OUTPUT / PARSED.
    """
    if prompt is None:
        prompt = select_prompts(2, prompt_id)

    if "{step1_json}" not in prompt:
        raise ValueError("prompt has no {step1_json} placeholder to fill")

    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f if l.strip()]
    data = [r for r in data if r.get("status") == "ok"]

    if ids:
        data = [r for r in data if r["id"] in ids]
    if limit:
        data = data[:limit]

    print(f"[+] {len(data)} records to process (status=ok from previous step)  from {input_path}")
    if not data:
        return output_path

    stats = {}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(data, 1):
            step1_output = row["parsed_output"]
            filled_prompt = prompt.replace(
                "{step1_json}", json.dumps(step1_output, ensure_ascii=False, indent=2))

            raw, parsed, status, n_tokens = run_model(
                model_path, filled_prompt, row,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=device)

            print(f"\n{'=' * 78}")
            print(f"[{i}/{len(data)}] {row['id']}  status={status}  tokens={n_tokens}")

            print(f"\n{'-' * 28} INPUT (step 1 analysis result) {'-' * 28}")
            print("original scene:", row.get("source_prompt"))
            print(json.dumps(step1_output, ensure_ascii=False, indent=2))

            print(f"\n{'-' * 26} PROMPT (full model input) {'-' * 26}")
            print(filled_prompt)

            print(f"\n{'-' * 28} OUTPUT (raw model output) {'-' * 28}")
            print(raw)

            print(f"\n{'-' * 30} PARSED (parsed JSON) {'-' * 30}")
            print(json.dumps(parsed, ensure_ascii=False, indent=2) if parsed
                  else f"(no JSON could be parsed, status={status})")

            result = {
                "id": row["id"],
                "category": row.get("category"),
                "risk_type": row.get("risk_type"),
                "variant": row.get("variant"),
                "rank": row.get("rank"),
                "suitability": row.get("suitability"),
                "atomicizability": row.get("atomicizability"),
                "source_prompt": row.get("source_prompt"),
                "prompt_id": prompt_id or "PROMPT",
                "model": model_path,
                "status": status,
                "n_tokens": n_tokens,
                "step1_output": step1_output,
                "raw_output": raw,
                "parsed_output": parsed,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            stats[status] = stats.get(status, 0) + 1

    print(f"\n{'=' * 78}")
    print("STEP2 done", dict(sorted(stats.items())))
    print("output", output_path)
    return output_path


def run_step_3(
    input_path="/home/ruomai/data/steps/step2_v2.jsonl",
    prompt=None,
    output_path="/home/ruomai/data/steps/step3_v2.jsonl",
    model_path="/mnt/hdd1/ruomai/models/Qwen3.5-9B-Base",
    prompt_id=None,
    device="cuda:1",
    ids=None,
    limit=None,
    max_new_tokens=2500,
    temperature=0.0,
):
    """Run step 3: decompose high-level elements into components that can be
    interpreted independently and benignly, and save to a jsonl file.

    {input_json} is fed the step 1 entities/relations/environment plus the step 2
    causality, combined into a single object.
    Each record prints four sections: INPUT / PROMPT / OUTPUT / PARSED.
    """
    if prompt is None:
        prompt = select_prompts(3, prompt_id)

    if "{input_json}" not in prompt:
        raise ValueError("prompt has no {input_json} placeholder to fill")

    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f if l.strip()]
    data = [r for r in data if r.get("status") == "ok"]

    if ids:
        data = [r for r in data if r["id"] in ids]
    if limit:
        data = data[:limit]

    print(f"[+] {len(data)} records to process (status=ok from previous step)  from {input_path}")
    if not data:
        return output_path

    stats = {}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(data, 1):
            step1_output = row.get("step1_output") or {}
            step2_output = row["parsed_output"]

            payload = {
                "key_semantic_elements": (step1_output.get("entities")
                                          or step1_output.get("key_semantic_elements")),
                "key_relational_factors": step1_output.get("key_relational_factors"),
                "environment": step1_output.get("environment"),
                "causal_semantics": step2_output,
            }
            for k in ("harm_analysis", "harm_relations"):
                if step1_output.get(k):
                    payload[k] = step1_output[k]

            filled_prompt = prompt.replace(
                "{input_json}", json.dumps(payload, ensure_ascii=False, indent=2))

            raw, parsed, status, n_tokens = run_model(
                model_path, filled_prompt, row,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=device)

            print(f"\n{'=' * 78}")
            print(f"[{i}/{len(data)}] {row['id']}  status={status}  tokens={n_tokens}")

            print(f"\n{'-' * 26} INPUT (merged step 1 + step 2 input) {'-' * 26}")
            print("original scene:", row.get("source_prompt"))
            print(json.dumps(payload, ensure_ascii=False, indent=2))

            print(f"\n{'-' * 26} PROMPT (full model input) {'-' * 26}")
            print(filled_prompt)

            print(f"\n{'-' * 28} OUTPUT (raw model output) {'-' * 28}")
            print(raw)

            print(f"\n{'-' * 30} PARSED (parsed JSON) {'-' * 30}")
            print(json.dumps(parsed, ensure_ascii=False, indent=2) if parsed
                  else f"(no JSON could be parsed, status={status})")

            result = {
                "id": row["id"],
                "category": row.get("category"),
                "risk_type": row.get("risk_type"),
                "variant": row.get("variant"),
                "rank": row.get("rank"),
                "suitability": row.get("suitability"),
                "atomicizability": row.get("atomicizability"),
                "source_prompt": row.get("source_prompt"),
                "prompt_id": prompt_id or "PROMPT",
                "model": model_path,
                "status": status,
                "n_tokens": n_tokens,
                "step1_output": step1_output,
                "step2_output": step2_output,
                "raw_output": raw,
                "parsed_output": parsed,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            stats[status] = stats.get(status, 0) + 1

    print(f"\n{'=' * 78}")
    print("STEP3 done", dict(sorted(stats.items())))
    print("output", output_path)
    return output_path


def run_step_4(
    input_path="/home/ruomai/data/steps/step3_v2.jsonl",
    prompt=None,
    output_path="/home/ruomai/data/steps/step4_v2.jsonl",
    model_path="/mnt/hdd1/ruomai/models/Qwen3.5-9B-Base",
    prompt_id=None,
    device="cuda:1",
    ids=None,
    limit=None,
    max_new_tokens=3000,
    temperature=0.0,
):
    """Run step 4: reconstruct the results of the first three steps into a complete
    3D scene and save to a jsonl file.

    The final product is final_scene_prompt, which is printed separately once this
    step finishes.
    Each record prints four sections: INPUT / PROMPT / OUTPUT / PARSED.
    """
    if prompt is None:
        prompt = select_prompts(4, prompt_id)

    for ph in ("{step1_json}", "{step2_json}", "{step3_json}"):
        if ph not in prompt:
            raise ValueError(f"prompt has no {ph} placeholder to fill")

    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f if l.strip()]
    data = [r for r in data if r.get("status") == "ok"]

    if ids:
        data = [r for r in data if r["id"] in ids]
    if limit:
        data = data[:limit]

    print(f"[+] {len(data)} records to process (status=ok from previous step)  from {input_path}")
    if not data:
        return output_path

    stats = {}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(data, 1):
            step1_output = row.get("step1_output") or {}
            step2_output = row.get("step2_output") or {}
            step3_output = row["parsed_output"]

            filled_prompt = (
                prompt
                .replace("{step1_json}", json.dumps(step1_output, ensure_ascii=False, indent=2))
                .replace("{step2_json}", json.dumps(step2_output, ensure_ascii=False, indent=2))
                .replace("{step3_json}", json.dumps(step3_output, ensure_ascii=False, indent=2))
            )

            raw, parsed, status, n_tokens = run_model(
                model_path, filled_prompt, row,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=device)

            final_scene_prompt = (parsed or {}).get("final_scene_prompt", "")

            print(f"\n{'=' * 78}")
            print(f"[{i}/{len(data)}] {row['id']}  status={status}  tokens={n_tokens}")

            print(f"\n{'-' * 26} INPUT (results of the first three steps) {'-' * 26}")
            print("original scene:", row.get("source_prompt"))
            print("step1:", json.dumps(step1_output, ensure_ascii=False)[:600], "...")
            print("step2:", json.dumps(step2_output, ensure_ascii=False)[:600], "...")
            print("step3:", json.dumps(step3_output, ensure_ascii=False)[:600], "...")

            print(f"\n{'-' * 26} PROMPT (full model input) {'-' * 26}")
            print(filled_prompt)

            print(f"\n{'-' * 28} OUTPUT (raw model output) {'-' * 28}")
            print(raw)

            print(f"\n{'-' * 30} PARSED (parsed JSON) {'-' * 30}")
            print(json.dumps(parsed, ensure_ascii=False, indent=2) if parsed
                  else f"(no JSON could be parsed, status={status})")

            if final_scene_prompt:
                print(f"\n{'*' * 78}")
                print("FINAL SCENE PROMPT (final product):")
                print(final_scene_prompt)
                print("*" * 78)

            result = {
                "id": row["id"],
                "category": row.get("category"),
                "risk_type": row.get("risk_type"),
                "variant": row.get("variant"),
                "rank": row.get("rank"),
                "suitability": row.get("suitability"),
                "atomicizability": row.get("atomicizability"),
                "source_prompt": row.get("source_prompt"),
                "prompt_id": prompt_id or "PROMPT",
                "model": model_path,
                "status": status,
                "n_tokens": n_tokens,
                "step1_output": step1_output,
                "step2_output": step2_output,
                "step3_output": step3_output,
                "raw_output": raw,
                "parsed_output": parsed,
                "final_scene_prompt": final_scene_prompt,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            stats[status] = stats.get(status, 0) + 1

    print(f"\n{'=' * 78}")
    print("STEP4 done", dict(sorted(stats.items())))
    print("output", output_path)
    return output_path


if __name__ == "__main__":
    run_step_1(input_path="/home/ruomai/data/i2p_claude_policy_flagged.json")
    run_step_1_5()
    run_step_1_7()
