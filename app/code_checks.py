"""第4回 宿題（実装）: LLM に聞かずにコードで判定する項目を、評価セットの各ケースに書いておき、まとめて判定する。

使い方:
    uv run python -m app.code_checks --cases data/eval/my_cases.jsonl
    uv run python -m app.code_checks --cases data/eval/my_cases.jsonl --labels lectures/04/my_labels.jsonl \
        --out lectures/04/my_code_checks.json

判定に使うのは、回答の文字列と、回答を生成したときのトレース（case["trace"]。app.judge --generate が残す）だけ。
API は呼ばない。判定は 3 種類:

- 全ケース共通:
    - lecture_mark   末尾に「第N回」があるか（app.judge と同じ関数）
    - glossary_lookup 参照に用語集のエントリがある質問で、search_glossary を呼んだか
      （INSTRUCTIONS の「答える前に、必ず search_glossary で用語集を確認する」）
- ケースごとに reference.code に書くもの:
    - {"type": "must_call", "value": "make_quiz"}                 そのツールを呼んだか
    - {"type": "must_not_contain", "value": ["保存しました"]}      どれも回答に含まないか
    - {"type": "must_contain_any", "value": ["用語集にはない", ...]} どれか 1 つを回答に含むか
  それぞれに "criterion"（関係する採点基準のキー。基準に無ければ null）と "why"（何を確かめるか）を書く。

--labels を付けると、criterion のある判定について、人のラベルとの食い違いを出す。
コードが fail で人が pass なら「コードが厳しすぎる（言い回しの違いを拾えない）」、
コードが pass で人が fail なら「コードでは見えない（文字列やツール呼び出しでは決まらない）」ことを示す。
"""

import argparse
import json
from pathlib import Path

from app.judge import NAMES, check_lecture_mark, load_labels, read_jsonl

ROOT = Path(__file__).resolve().parent.parent


def _check(name: str, criterion: str | None, ok: bool | None, detail: str) -> dict:
    verdict = "skip" if ok is None else ("pass" if ok else "fail")
    return {"check": name, "criterion": criterion, "verdict": verdict, "detail": detail}


def run_checks(case: dict) -> list[dict]:
    answer = case.get("answer", "")
    trace = case.get("trace")
    calls = trace["tool_calls"] if trace else None
    ref = case.get("reference", {})
    out = [_check("lecture_mark", "lecture_mark", check_lecture_mark(answer) == "pass", "末尾の行に「第N回」がある")]

    if ref.get("glossary"):
        ok = None if calls is None else ("search_glossary" in calls)
        out.append(_check("glossary_lookup", "grounding", ok, "用語集にエントリがある質問で search_glossary を呼んだ"))

    for spec in ref.get("code", []):
        kind, value = spec["type"], spec["value"]
        values = value if isinstance(value, list) else [value]
        if kind == "must_call":
            ok = None if calls is None else all(v in calls for v in values)
        elif kind == "must_not_contain":
            ok = not any(v in answer for v in values)
        elif kind == "must_contain_any":
            ok = any(v in answer for v in values)
        else:
            raise ValueError(f"未知の判定の種類: {kind}")
        out.append(_check(f"{kind}:{'/'.join(values)}", spec.get("criterion"), ok, spec.get("why", "")))
    return out


def compare_with_labels(results: dict[str, list[dict]], labels: dict[str, dict]) -> list[dict]:
    """criterion のある判定について、人のラベルと食い違ったものを返す。"""
    diffs = []
    for cid, checks in results.items():
        if cid not in labels:
            continue
        for c in checks:
            human = labels[cid].get(c["criterion"]) if c["criterion"] else None
            if c["verdict"] == "skip" or human not in ("pass", "fail") or c["verdict"] == human:
                continue
            kind = "コードが厳しすぎる" if c["verdict"] == "fail" else "コードでは見えない"
            diffs.append({"id": cid, "check": c["check"], "criterion": c["criterion"],
                          "code": c["verdict"], "human": human, "kind": kind})
    return diffs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="評価セットの各ケースに書いたコード判定をまとめて実行する（API 不要）")
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "eval" / "my_cases.jsonl")
    parser.add_argument("--labels", type=Path, help="人のラベル（jsonl）。付けると食い違いを出す")
    parser.add_argument("--out", type=Path, help="結果の保存先（json）")
    args = parser.parse_args(argv)

    cases = read_jsonl(args.cases)
    if not any(c.get("trace") for c in cases):
        print("注意: trace がありません。先に app.judge --generate で回答を生成すると、ツール呼び出しの判定もできます")
    results = {c["id"]: run_checks(c) for c in cases}

    n_pass = sum(c["verdict"] == "pass" for cs in results.values() for c in cs)
    n_fail = sum(c["verdict"] == "fail" for cs in results.values() for c in cs)
    print(f"コード判定 {len(cases)} 件（pass {n_pass} / fail {n_fail}）")
    for case in cases:
        checks = results[case["id"]]
        marks = "  ".join(f"{'○' if c['verdict'] == 'pass' else ('×' if c['verdict'] == 'fail' else '-')} {c['check']}" for c in checks)
        print(f"  #{case['id']} {case.get('category', '')}: {marks}")

    diffs = []
    if args.labels:
        diffs = compare_with_labels(results, load_labels(args.labels))
        print(f"\n人のラベルとの食い違い {len(diffs)} 件")
        for d in diffs:
            print(f"  #{d['id']} {d['check']}（{NAMES.get(d['criterion'], d['criterion'])}） コード={d['code']} 人={d['human']} → {d['kind']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"results": results, "label_diffs": diffs}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n→ {args.out} に保存しました")


if __name__ == "__main__":
    main()
