import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

with open("data/benchmark_dev_70.json", encoding="utf-8") as f:
    dev = json.load(f)

with open("ket_qua_eval_dev.txt", encoding="utf-8") as f:
    eval_text = f.read()

eval_map = {}
for block in eval_text.split("[dev_")[1:]:
    header = block.split("\n")[0]
    qid = "dev_" + header.split("]")[0]
    eval_map[qid] = block

out_lines = ["--- FAILING DEV QUERIES ANALYSIS ---"]
for item in dev:
    qid = item["id"]
    block = eval_map.get(qid, "")
    if "R@5=0.0" in block or "R@1=0.0" in block:
        out_lines.append(f"\nID: {qid} | Expected Doc: {item['expected_doc']}")
        out_lines.append(f"Query: {item['query']}")
        for line in block.split("\n")[:4]:
            out_lines.append(f"  {line}")

output_text = "\n".join(out_lines)
Path("scratch").mkdir(exist_ok=True)
with open("scratch/diagnose_output.txt", "w", encoding="utf-8") as f:
    f.write(output_text)

print(f"Written {len(out_lines)} lines to scratch/diagnose_output.txt")
