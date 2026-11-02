#!/usr/bin/env python
"""第2回 宿題（基礎）: run_lora.py の result.json を 2 つ並べ、比較表と answer.md の下書きを作る。

    python compare.py --baseline /content/qlora_lab/hw/baseline --variant /content/qlora_lab/hw/epochs5 --out /content/qlora_lab/hw

出力
- comparison.md : 比較表と、学習後に落ちた質問の一覧（失敗の型つき）
- answer.md     : lectures/02/answer.md にそのまま貼れる下書き（演習欄は元の設定の実行結果から作る）
考察の文は数字に応じて組み立てているが、最後は出力原文を自分で読んで直すこと。
"""
import argparse
import json
from pathlib import Path

FAQ_SENTENCES = {
    "課題の提出期限は 9 月 30 日 23:59（日本時間）です。",
    "提出先は学習ポータルの「課題」欄です。",
    "提出形式は PDF 1 ファイルで、ファイル名は 学籍番号_課題番号.pdf です。",
    "質問は毎週火曜 17:00〜18:00 のオフィスアワーで受け付けます。",
    "期限後の提出は、期限から 48 時間以内なら減点のうえ受け付けます。",
    "この FAQ には記載がありません。運営に確認が必要です。",
}
CONTROL_Q = "課題の提出期限はいつですか"   # 知識質問のうち FAQ にある対照（ノートブックの "FAQ にある" 列は False と出るが、FAQ にある）
KEY_JA = {"num_train_epochs": "num_train_epochs（エポック数）", "lr": "lr（学習率）", "lora_r": "lora_r（LoRA の次元）"}

ap = argparse.ArgumentParser()
ap.add_argument("--baseline", required=True)
ap.add_argument("--variant", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()


def load(d):
    d = Path(d)
    res = json.loads((d / "result.json").read_text(encoding="utf-8"))
    after = json.loads((d / "eval_after.json").read_text(encoding="utf-8"))["rows"]
    before = json.loads((d / "eval_before.json").read_text(encoding="utf-8"))["rows"]
    return res, before, after


B, B_before, B_after = load(args.baseline)
V, V_before, V_after = load(args.variant)
assert B["changed"] == {}, "baseline は元の設定で実行したものを指定する"
assert len(V["changed"]) == 1, "variant は 1 項目だけ変えたものを指定する"
KEY, VAL = next(iter(V["changed"].items()))
BVAL = B["config"][KEY]
N = B["after"]["n"]


def rate(x, n=N):
    return f"{x:.3f}（{round(x * n)}/{n}）"


def f4(x):
    return "—" if x is None else f"{x:.4f}"


def failure_type(r):
    if r["both_ok"]: return None
    if not r["format_ok"]: return f"形式の崩れ（{r['format_reason']}）"
    obj = json.loads(r["raw"])
    if r["gold_needs_review"] and obj["needs_review"] is False: return "根拠なしの質問に断定"
    if not r["gold_needs_review"] and obj["needs_review"] is True: return "根拠ありの質問に確認要求で逃げた"
    return "needs_review は正しいが必須語句が不足"


def answer_of(r):
    try: return json.loads(r["raw"]).get("answer")
    except Exception: return None


def failures(rows):
    return [(r["question"], failure_type(r), r["raw"]) for r in rows if not r["both_ok"]]


def verbatim_share(rows):
    ok = [r for r in rows if r["format_ok"]]
    return sum(answer_of(r) in FAQ_SENTENCES for r in ok), len(ok)


def knowledge_line(res):
    return " / ".join(f"「{k['質問']}」{k['学習前の分類']} → {k['学習後の分類']}" for k in res["knowledge"])


def nonfaq_asserted(res):
    return [k for k in res["knowledge"] if k["質問"] != CONTROL_Q and k["学習後の分類"] == "asserted"]


def fail_md(fs):
    if not fs: return "なし"
    return "<br>".join(f"「{q}」: {t} — `{raw}`" for q, t, raw in fs)


tb, tv = B["train"], V["train"]

# ------------------------------------------------------------ 比較表
rows = [
    ("変えた設定", f"{KEY} = {BVAL}", f"{KEY} = {VAL}"),
    ("学習ステップ数 / エポック", f"{tb['global_step']} / {tb['epoch']:.1f}", f"{tv['global_step']} / {tv['epoch']:.1f}"),
    ("学習前 format_rate / content_rate", f"{rate(B['before']['format_rate'])} / {rate(B['before']['content_rate'])}",
     f"{rate(V['before']['format_rate'])} / {rate(V['before']['content_rate'])}"),
    ("**学習後 format_rate**", rate(B["after"]["format_rate"]), rate(V["after"]["format_rate"])),
    ("**学習後 content_rate**", rate(B["after"]["content_rate"]), rate(V["after"]["content_rate"])),
    ("学習後 both_rate", rate(B["after"]["both_rate"]), rate(V["after"]["both_rate"])),
    ("退行した質問", str(len(B["regressed"])), str(len(V["regressed"]))),
    ("**最終 loss**（最終ステップ）", f4(tb["final_loss"]), f4(tv["final_loss"])),
    ("最後の 10 ステップの loss 平均", f4(tb["final_loss_last10_mean"]), f4(tv["final_loss_last10_mean"])),
    ("全ステップの loss 平均", f4(tb["mean_train_loss"]), f4(tv["mean_train_loss"])),
    ("学習後の valid loss（7 件）", f4(tb["valid_loss"]), f4(tv["valid_loss"])),
    ("**学習時間（秒）**", f"{tb['train_seconds']:.1f}", f"{tv['train_seconds']:.1f}"),
    ("trainable%", f"{tb['trainable_pct']:.3f}", f"{tv['trainable_pct']:.3f}"),
    ("知識質問（学習前 → 学習後）", knowledge_line(B), knowledge_line(V)),
]
table = "| 指標 | 元の設定（baseline） | 変更後（" + V["label"] + "） |\n| --- | --- | --- |\n" + \
        "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows)

diff_q = []
for b, v in zip(B_after, V_after):
    assert b["question"] == v["question"]
    if b["both_ok"] != v["both_ok"]:
        diff_q.append(f"- 「{b['question']}」: baseline {'○' if b['both_ok'] else '×（' + failure_type(b) + '）'} / "
                      f"{V['label']} {'○' if v['both_ok'] else '×（' + failure_type(v) + '）'}")
diff_md = "\n".join(diff_q) if diff_q else "- なし（どちらの設定でも同じ質問が○／×になった）"

comparison = f"""# 宿題（基礎）比較: {KEY} {BVAL} → {VAL}

{table}

## 学習後に落ちた質問（失敗の型）

- baseline: {fail_md(failures(B_after))}
- {V['label']}: {fail_md(failures(V_after))}

## 2 つの設定で結果が分かれた質問

{diff_md}

## loss の推移（step: loss）

- baseline: {", ".join(f"{e['step']}:{e['loss']:.3f}" for e in tb["loss_log"])}
- {V['label']}: {", ".join(f"{e['step']}:{e['loss']:.3f}" for e in tv["loss_log"])}

実行環境: {B.get('device')} / {json.dumps(B['versions'], ensure_ascii=False)}
"""

# ------------------------------------------------------------ 考察（数字に応じて組み立てる）
fb, fv = B["after"]["format_rate"], V["after"]["format_rate"]
cb, cv = B["after"]["content_rate"], V["after"]["content_rate"]
sb, sv = tb["global_step"], tv["global_step"]
desc = f"{KEY} を {BVAL} から {VAL} に変える（学習ステップ {sb} → {sv}）と"
s = []
if abs(fv - fb) < 1e-9:
    s.append(f"{desc}、学習後の形式遵守率はどちらも {fv:.3f} で差は出なかった。")
    if sv < sb and fv == 1.0:
        s.append(f"JSON の形式は {sv} ステップの学習でも完全に身についた。")
else:
    s.append(f"{desc}、学習後の形式遵守率は {fb:.3f} → {fv:.3f} に{'下がった' if fv < fb else '上がった'}。")
    if fv < fb and sv < sb:
        s.append("形式の習得にも一定の学習量が要る。")
dc = round((cv - cb) * N)
if dc == 0:
    s.append(f"内容正解率もどちらも {cv:.3f} だった。")
else:
    s.append(f"内容正解率は {cb:.3f} → {cv:.3f}（{abs(dc)} 問{'減' if dc < 0 else '増'}）だった。")
    if abs(dc) <= 2:
        s.append(f"ただし 1 問は約 {100 / N:.1f} ポイントに当たり、21 問・1 回の実行（GPU 上の学習は seed を固定しても完全には再現しない）では、1〜2 問の差は誤差の範囲として扱うべきである。")
s.append(f"最終ステップの loss は {f4(tb['final_loss'])} → {f4(tv['final_loss'])}、valid loss は {f4(tb['valid_loss'])} → {f4(tv['valid_loss'])}、"
         f"学習時間は {tb['train_seconds']:.0f} 秒 → {tv['train_seconds']:.0f} 秒だった。")
if tb["final_loss"] is not None and tv["final_loss"] is not None:
    lo_name, lo_c, hi_c = ("元の設定", cb, cv) if tb["final_loss"] < tv["final_loss"] else (V["label"], cv, cb)
    if lo_c <= hi_c:
        s.append(f"train loss が低い方（{lo_name}）の内容正解率は、もう一方と{'同じ' if lo_c == hi_c else 'くらべてむしろ低い'}だった。"
                 "loss は教師文をどれだけ再現できるかの指標であり、FAQ に根拠があるかの判断の正しさまでは表さない。")
if KEY == "num_train_epochs":
    w = B["config"]["warmup_steps"]
    s.append(f"なお warmup_steps={w} はステップ数で固定されているため、{sv} ステップの学習では warmup が全体の {100 * min(w, sv) / sv:.0f}%（元の設定では {100 * w / sb:.0f}%）を占める。"
             "エポック数の変更は、更新回数だけでなく学習率スケジュールの形も変えている。")
    if sb and sv:
        s.append(f"学習時間はステップ数の比（{sv / sb:.2f}）に対して {tv['train_seconds'] / tb['train_seconds']:.2f} 倍で、ほぼステップ数に比例する。")
if KEY == "lora_r":
    s.append(f"trainable% は {tb['trainable_pct']:.3f} → {tv['trainable_pct']:.3f} と変わったが、学習時間の比は {tv['train_seconds'] / tb['train_seconds']:.2f} 倍だった。"
             "lora_alpha は 32 のまま固定したので、adapter 出力の倍率 alpha/r も同時に変わっている。")
p_compare = "".join(s)

k_share, k_n = verbatim_share(B_after)
asserted_test = [(q, raw) for q, t, raw in failures(B_after) + failures(V_after) if t == "根拠なしの質問に断定"]
asserted_know = nonfaq_asserted(B) + nonfaq_asserted(V)
fb0 = B["before"]["format_rate"]
if fb > fb0:
    p_good = (f"ファインチューニングが向くのは、出力の形式・文体・定型の振る舞いを揃えたい場面である。今回、プロンプトで JSON 形式を明示しても学習前の形式遵守率は "
              f"{fb0:.3f} だったが、全パラメータの {tb['trainable_pct']:.2f}% だけを学習する LoRA で {fb:.3f} になった。"
              "形式の要件をプロンプトに長く書かずに済み、推論 1 回あたりのコストも増えない。出力を後段のプログラムが機械的に読む場面（JSON、分類ラベル、定型の確認メッセージ）では効果が大きい。")
else:
    p_good = (f"ファインチューニングは出力形式を揃える用途で使われることが多いが、今回は学習後も形式遵守率が {fb0:.3f} → {fb:.3f} で改善せず、"
              "この用途での効果は今回の結果からは確かめられなかった。")
if asserted_test or asserted_know:
    if asserted_know:
        ex_q, ex_a = asserted_know[0]["質問"], asserted_know[0]["学習後の出力"]
    else:
        ex_q, ex_a = asserted_test[0]
    evid = (f"FAQ に無い質問に、FAQ の別項目を流用して needs_review: false で断定する出力が残った（例:「{ex_q}」→ `{ex_a}`）。"
            "質問の語が FAQ の項目と表面上似ているかで答えを選んでいるように見え、「根拠があるか」という判断そのものが身についたとは言えない（今回の出力から見た仮説で、検証はしていない）。")
else:
    evid = ("今回の test と知識質問では断定は見られなかったが、根拠なしの test は 6 問、知識質問は 3 問しかなく、判断が身についたと結論するには少なすぎる。")
p_bad = (f"向かないのは、新しい知識を覚えさせたい場面や、事実が頻繁に変わる場面である。学習後の回答は、形式を守れた出力のうち "
         f"{k_share}/{k_n} 件が教師データの文（FAQ 原文か確認要求の定型文）とまったく同じで、学習したのは主に「どの定型文を返すか」だった。{evid}"
         "また FAQ の内容（締切など）が変わるたびに再学習が必要で、答えがどの文書に基づくかも示せない。"
         "事実を答える部分は検索（RAG）で根拠を入力に渡して更新を容易にし、ファインチューニングは出力形式と「根拠が無いときは確認を求める」振る舞いの補強に使う、という分担が妥当である。")

# ------------------------------------------------------------ 演習欄（元の設定の実行から）
c = B["config"]
cfg_line = (f"`{c['base_model']}`（revision `{str(c['revision'])[:7]}`）、LoRA（量子化なし）、lora_r={c['lora_r']}、lora_alpha={c['lora_alpha']}、"
            f"lr={c['lr']}、num_train_epochs={c['num_train_epochs']}（{tb['global_step']} ステップ）、warmup_steps={c['warmup_steps']}、seed={c['seed']}、"
            f"貪欲生成（do_sample=False）、{B.get('device')}")
b_fail = failures(B_after)
b_fail_txt = "なし" if not b_fail else "；".join(f"「{q}」→ {t}（`{raw}`）" for q, t, raw in b_fail)
changed_txt = (f"形式遵守率が {B['before']['format_rate']:.3f} → {fb:.3f}、内容正解率が {B['before']['content_rate']:.3f} → {cb:.3f} に上がった。"
               f"学習後の answer は FAQ 原文をほぼそのまま返す形になった（形式を守れた出力の {k_share}/{k_n} 件が教師データの文と一致）。")
unchanged_txt = ("FAQ に根拠があるかの判断は保証されなかった。" + ("根拠なしの質問や知識質問で、FAQ の別項目を当てはめて断定する出力が残った。"
                 if (any(t == "根拠なしの質問に断定" for _, t, _ in b_fail) or nonfaq_asserted(B)) else
                 "今回は test・知識質問とも断定は出なかったが、根拠なしの問題数が少なく、判断が身についたとまでは言えない。"))

answer = f"""# 第2回 解答

## 演習

- 使ったモデル・設定（`result.json` の `config` から）：{cfg_line}
- 自分で学習を実行したか（`executed_training_myself`）：はい（実行できた範囲：セクション 0〜7 と同じ処理を Colab の T4 で実行。宿題の比較と条件を揃えるため、ノートブックと同じ処理を 1 プロセスで行う `homework/run_lora.py` で実行した）
- 学習前：`format_rate` = {rate(B['before']['format_rate'])} / `content_rate` = {rate(B['before']['content_rate'])}
- 学習後：`format_rate` = {rate(fb)} / `content_rate` = {rate(cb)}
- 退行した質問：{'、'.join(B['regressed']) if B['regressed'] else 'なし'}
- 学習後に残った失敗と、その型（根拠なしの質問に断定した / 根拠ありの質問に確認要求で逃げた / 形式の崩れ）：{b_fail_txt}
- 知識を問う質問（セクション 6）の結果（学習前の分類 → 学習後の分類）と気づいたこと：{knowledge_line(B)}。対照の「{CONTROL_Q}」は FAQ にある質問（ノートブックの「FAQ にある」列は `q in gold` で判定しているため False と表示されるが、train に含まれる質問である）。
- 学習で変わったこと・変わらなかったこと：{changed_txt}{unchanged_txt}

## 宿題

- 取り組んだレベル：基礎
- 変えた設定やデータ：`{KEY}` を {BVAL} → {VAL} に変更（学習ステップ {sb} → {sv}）。その他の設定・データ・評価は元の設定と同じ。設定ごとに別プロセスで上から実行し（セッション再起動と同じ条件）、`result.json` を 2 つ比べた。最終 loss を正確に取るため loss を毎ステップ記録した（`logging_steps=1`。ログの間隔だけの変更で学習には影響しない）。結果ファイルは `lectures/02/homework/results/` に置いた
- 結果（比較表や数字）：

{table}

  2 つの設定で結果が分かれた質問：

{diff_md}

- ファインチューニングが向く場面・向かない場面についての考察：
  - 今回の比較から：{p_compare}
  - {p_good}
  - {p_bad}
  - AI ツールの利用：比較用スクリプトと answer.md の下書きの作成に AI（Claude）を使った。数値は自分の Colab 実行の `result.json` から自動で転記し、`eval_after.json` の出力原文を読んで失敗の型を確認した。
"""

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / "comparison.md").write_text(comparison, encoding="utf-8")
(out / "answer.md").write_text(answer, encoding="utf-8")
print(comparison)
print("書き出し:", out / "comparison.md", out / "answer.md")
