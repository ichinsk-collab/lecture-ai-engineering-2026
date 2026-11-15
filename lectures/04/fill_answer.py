"""第4回 宿題: my_judge_result.json の結果を answer.md の印（<!--JUDGE_SUMMARY--> など）に書き込む。

    uv run python lectures/04/fill_answer.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
result = json.loads((HERE / "my_judge_result.json").read_text(encoding="utf-8"))
s = result["summary"]

per = "、".join(f"{p['name']} {p['agree']}/{p['compared']}" for p in s["per_criterion"].values())
o, c = s["overall"], s["coverage"]
summary = (f"採点モデル {result['model']}。基準別の一致率は {per}。総合の一致率 {o['agree']}/{o['compared']}"
           f"（自動判定できた {c['covered']}/{c['total']} 件）。要確認 {len(s['review'])} 件")

items = [f"#{r['id']} {r['name']}（要確認）：{r['reason']}" for r in s["review"]]
items += [f"#{m['id']} {m['name']}（{m['kind']}：LLM={m['judge']}、ラベル={m['human']}）：{m['reason']}" for m in s["mismatches"]]
mismatch = "なし" if not items else "\n" + "\n".join(f"    - {x}" for x in items)

path = HERE / "answer.md"
text = path.read_text(encoding="utf-8")
assert "<!--JUDGE_SUMMARY-->" in text, "answer.md の印がありません（すでに書き込み済みかもしれません）"
path.write_text(text.replace("<!--JUDGE_SUMMARY-->", summary).replace("<!--JUDGE_MISMATCH-->", mismatch), encoding="utf-8")
print(summary)
print("不一致・要確認:", mismatch)
