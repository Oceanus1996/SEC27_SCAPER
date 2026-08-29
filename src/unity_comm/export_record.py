"""record_i2p_filled.xlsx  ->  io/record_prompts.jsonl
After editing the Excel file, run this, then click "Reload table" in the Box Pipeline window.
  python export_record.py
"""
import json, os, openpyxl

SRC = r"C:\Users\rrm_a\OneDrive\文档\record_i2p_filled.xlsx"
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "io", "record_prompts.jsonl")

def main():
    wb = openpyxl.load_workbook(SRC); ws = wb.active
    rows = []
    for r in range(2, ws.max_row + 1):
        boxnum = ws.cell(r, 1).value
        _id = ws.cell(r, 2).value
        label = ws.cell(r, 3).value
        steps = []
        for c in range(8, ws.max_column + 1):
            v = ws.cell(r, c).value
            if v not in (None, "") and str(v).strip():
                steps.append(str(v).strip())
        if boxnum is None and not _id and not steps:
            continue
        rows.append({"excel_row": r, "boxnum": boxnum, "id": _id, "label": label, "steps": steps})
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    with open(DST, "w", encoding="utf-8") as f:
        for o in rows:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {DST}")

if __name__ == "__main__":
    main()
