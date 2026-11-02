#!/usr/bin/env python
"""第2回 宿題（基礎）: CONFIG の 1 項目だけを変えて LoRA で再学習し、result.json を作る。

notebooks/lora_format_tuning.ipynb のセクション 1〜7 と同じ処理（データ生成・プロンプト・採点・学習設定）を
1 つのプロセスで実行する。1 回の実行が 1 つの設定に対応し、プロセスが終わると GPU 上のモデルも消えるので、
「ランタイム → セッションを再起動して上から実行」と同じ条件で比べられる。

ノートブックから変えた点（学習そのものは変えていない）
- 宿題で記録する「最終 loss」と「学習時間」を result.json の "train" に記録する。
- loss を毎ステップ記録する（logging_steps=1）。ノートブックの 10 ステップごとの記録だと、学習が 10 ステップしかない
  設定では最終 loss が「1〜10 ステップの平均」になり、元の設定（31〜40 ステップの平均）と比べられないため。
  logging_steps はログの間隔だけを決める設定で、勾配や学習率には影響しない。
- 学習後に valid（7 件）の loss を 1 回だけ測る（trainer.evaluate()）。学習には使わない。

使い方（Colab の T4 で）
    python run_lora.py --label baseline                          # 元の設定（20 エポック）
    python run_lora.py --label epochs5 --num_train_epochs 5      # 1 項目だけ変える
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

# ---------------------------------------------------------------- 1. 設定
BASE_CONFIG = {
    "base_model": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "seed": 42,
    "num_train_epochs": 20,
    "warmup_steps": 6,
    "lr": 2e-4,
    "lora_r": 16,
    "lora_alpha": 32,
    "max_seq_len": 512,
    "max_new_tokens": 96,
    "gen": {"do_sample": False},
}
HW_KEYS = ("lora_r", "lr", "num_train_epochs")  # 宿題（基礎）で変えてよい項目

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True, help="結果フォルダの名前（例: baseline, epochs5）")
ap.add_argument("--lora_r", type=int)
ap.add_argument("--lr", type=float)
ap.add_argument("--num_train_epochs", type=int)
ap.add_argument("--out_root", default="/content/qlora_lab/hw")
# 以下は動作確認用。宿題では既定値のまま使う
ap.add_argument("--base_model", default=BASE_CONFIG["base_model"])
ap.add_argument("--revision", default=BASE_CONFIG["revision"], help="'none' で revision を指定しない")
ap.add_argument("--max_new_tokens", type=int, default=BASE_CONFIG["max_new_tokens"])
args = ap.parse_args()

overrides = {k: getattr(args, k) for k in HW_KEYS if getattr(args, k) is not None}
assert len(overrides) <= 1, f"宿題（基礎）は 1 項目だけ変える: {overrides}"
CONFIG = dict(BASE_CONFIG, **overrides)
CONFIG["base_model"] = args.base_model
CONFIG["revision"] = None if args.revision.lower() == "none" else args.revision
CONFIG["max_new_tokens"] = args.max_new_tokens
# 注: lora_r を変えても lora_alpha は 32 のまま（1 項目だけ変える）。adapter 出力の倍率 alpha/r も同時に変わる

WORK = Path(args.out_root) / args.label
WORK.mkdir(parents=True, exist_ok=True)
print("label:", args.label, "| 変えた設定:", overrides or "なし（元の設定）")
print(json.dumps(CONFIG, ensure_ascii=False, indent=2))

# ---------------------------------------------------------------- 2. 架空 FAQ と教師データ（ノートブックと同一）
FAQ = {
    "deadline": "課題の提出期限は 9 月 30 日 23:59（日本時間）です。",
    "destination": "提出先は学習ポータルの「課題」欄です。",
    "format": "提出形式は PDF 1 ファイルで、ファイル名は 学籍番号_課題番号.pdf です。",
    "office_hours": "質問は毎週火曜 17:00〜18:00 のオフィスアワーで受け付けます。",
    "late": "期限後の提出は、期限から 48 時間以内なら減点のうえ受け付けます。",
}
FAQ_TEXT = "\n".join(f"- {v}" for v in FAQ.values())
GROUNDED = {
    "deadline": (["課題の提出期限はいつですか", "いつまでに出せばいいですか", "締切を教えてください", "提出の期限は？"], ["9 月 30 日", "23:59"]),
    "destination": (["提出先はどこですか", "どこにアップロードしますか", "課題はどこから出しますか", "提出場所は？"], ["学習ポータル", "課題"]),
    "format": (["提出形式を教えてください", "ファイル名の決まりはありますか", "何のファイル形式で出しますか", "PDF 以外でも出せますか"], ["PDF"]),
    "office_hours": (["質問はいつできますか", "オフィスアワーはいつですか", "先生に質問する時間はありますか", "相談できる時間帯は？"], ["火曜", "17:00"]),
    "late": (["遅れて出したらどうなりますか", "期限を過ぎた場合は受け付けますか", "遅延提出のルールは？", "締切後の扱いを教えてください"], ["48 時間", "減点"]),
}
UNGROUNDED = [
    "再提出はできますか", "欠席した回の扱いを教えてください", "他の受講者の点数は見られますか",
    "成績はいつ出ますか", "提出期限は延長できますか", "グループでの提出は可能ですか",
    "課題の配点はどうなっていますか", "講義の録画は公開されますか",
]
TEST_EXTRA_GROUNDED = {
    "deadline": ["提出期限の日付と時刻を教えてください", "課題の締切は何月何日の何時ですか"],
    "destination": ["課題の提出はどこで行いますか", "提出物を出す先を教えてください"],
    "format": ["提出するファイルの形式は何ですか", "提出ファイルは何形式で、いくつ出しますか"],
    "office_hours": ["オフィスアワーの曜日と時間を教えてください", "質問を受け付けてもらえるのは何曜日の何時からですか"],
    "late": ["提出が締切に間に合わなかったらどうなりますか", "期限後に提出した課題は受理されますか"],
}
TEST_EXTRA_UNGROUNDED = [
    "課題の文字数に制限はありますか", "提出した課題へのフィードバックはもらえますか",
    "参考文献の書き方に指定はありますか", "オフィスアワーは事前予約が必要ですか",
]
UNGROUNDED_ANSWER = "この FAQ には記載がありません。運営に確認が必要です。"


def make_example(question, answer, needs_review):
    target = json.dumps({"answer": answer, "needs_review": needs_review}, ensure_ascii=False)
    return {"question": question, "faq": FAQ_TEXT, "target": target, "needs_review": needs_review}


def split_templates(templates):
    return templates[:2], templates[2:3], templates[3:4]


train, valid, test, gold = [], [], [], {}
for key, (templates, must_include) in GROUNDED.items():
    tr, va, te = split_templates(templates)
    for q in tr: train.append(make_example(q, FAQ[key], False))
    for q in va: valid.append(make_example(q, FAQ[key], False))
    for q in te + TEST_EXTRA_GROUNDED[key]:
        test.append(make_example(q, FAQ[key], False)); gold[q] = {"needs_review": False, "must_include": must_include}
for i, q in enumerate(UNGROUNDED):
    ex = make_example(q, UNGROUNDED_ANSWER, True)
    bucket = [train, train, valid, test][i % 4]
    bucket.append(ex)
    if bucket is test: gold[q] = {"needs_review": True, "must_include": ["確認"]}
for q in TEST_EXTRA_UNGROUNDED:
    test.append(make_example(q, UNGROUNDED_ANSWER, True)); gold[q] = {"needs_review": True, "must_include": ["確認"]}

all_questions = [r["question"] for r in train + valid + test]
assert len(set(all_questions)) == len(all_questions)
assert (len(train), len(valid), len(test), len(gold)) == (14, 7, 21, 21)
print(f"train={len(train)} valid={len(valid)} test={len(test)} gold={len(gold)}")

# ---------------------------------------------------------------- 3. モデルの読み込みと学習前評価
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(CONFIG["base_model"], revision=CONFIG["revision"])
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(CONFIG["base_model"], revision=CONFIG["revision"], dtype=torch.float32).to(DEVICE)
model.config.use_cache = True
BASE_MEMORY_GB = round(model.get_memory_footprint() / 1e9, 2)
print("loaded:", CONFIG["base_model"], "| device:", DEVICE, "| 重みのメモリ(GB):", BASE_MEMORY_GB)

SYSTEM = "あなたは講座 FAQ の回答係です。与えられた FAQ だけを根拠に、必ず JSON 1 行で答えてください。形式: {\"answer\": 文字列, \"needs_review\": 真偽値}。FAQ に記載が無いことは answer に『確認が必要』と書き、needs_review を true にします。"


def build_messages(example):
    user = f"FAQ:\n{example['faq']}\n\n質問: {example['question']}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def generate(model, example):
    prompt = tok.apply_chat_template(build_messages(example), tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=CONFIG["max_new_tokens"], pad_token_id=tok.pad_token_id, **CONFIG["gen"])
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def check_format(text):
    try:
        obj = json.loads(text)
    except Exception as e:
        return False, f"json_error: {type(e).__name__}", None
    if not isinstance(obj, dict): return False, "not_object", None
    if set(obj) - {"answer", "needs_review"}: return False, "extra_keys", obj
    if "answer" not in obj or "needs_review" not in obj: return False, "missing_key", obj
    if not isinstance(obj["answer"], str) or not isinstance(obj["needs_review"], bool): return False, "wrong_type", obj
    return True, "", obj


def check_content(obj, gold_entry):
    if obj is None: return False
    if obj.get("needs_review") != gold_entry["needs_review"]: return False
    return all(w in obj.get("answer", "") for w in gold_entry["must_include"])


def mark(ok):
    return "○" if ok else "×"


def evaluate(model, label):
    rows = []
    t0 = time.time()
    for ex in test:
        raw = generate(model, ex)
        ok_fmt, reason, obj = check_format(raw)
        loose = obj
        if not ok_fmt:
            m = re.search(r"\{.*\}", raw, re.S)
            try: loose = json.loads(m.group(0)) if m else None
            except Exception: loose = None
        ok_content = check_content(loose if isinstance(loose, dict) else None, gold[ex["question"]])
        rows.append({"question": ex["question"], "raw": raw, "format_ok": ok_fmt, "format_reason": reason,
                     "content_ok": ok_content, "both_ok": ok_fmt and ok_content, "gold_needs_review": gold[ex["question"]]["needs_review"]})
    n = len(rows)
    summary = {"label": label, "n": n,
               "format_rate": sum(r["format_ok"] for r in rows) / n,
               "content_rate": sum(r["content_ok"] for r in rows) / n,
               "both_rate": sum(r["both_ok"] for r in rows) / n,
               "seconds": round(time.time() - t0, 1)}
    (WORK / f"eval_{label}.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    for r in rows:
        print(f"  [{mark(r['format_ok'])}/{mark(r['content_ok'])}] gold_nr={r['gold_needs_review']!s:5} {r['question']} -> {r['raw'][:90]}")
    return summary, rows


print("\n=== 学習前評価 ===")
before_summary, before_rows = evaluate(model, "before")

# ---------------------------------------------------------------- 4. LoRA 学習
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import Dataset


def to_chat(example):
    return {"messages": build_messages(example) + [{"role": "assistant", "content": example["target"]}]}


train_ds = Dataset.from_list([to_chat(e) for e in train])
valid_ds = Dataset.from_list([to_chat(e) for e in valid])

model.config.use_cache = False
lora = LoraConfig(r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
USE_AMP = False
sft = SFTConfig(output_dir=str(WORK / "ckpt"), num_train_epochs=CONFIG["num_train_epochs"], per_device_train_batch_size=4, gradient_accumulation_steps=2,
                learning_rate=CONFIG["lr"], lr_scheduler_type="cosine", warmup_steps=CONFIG["warmup_steps"],
                logging_steps=1,  # ノートブックは 10。ログの間隔だけの違いで学習には影響しない（冒頭の説明を参照）
                save_steps=10, save_total_limit=2,
                fp16=USE_AMP, max_length=CONFIG["max_seq_len"], packing=False, report_to="none", seed=CONFIG["seed"],
                # GPU が無いときの動作確認用。CUDA（T4）ではノートブックと同じく何も渡さない
                **({"use_cpu": True, "bf16": False} if DEVICE == "cpu" else {}))
trainer = SFTTrainer(model=model, args=sft, train_dataset=train_ds, eval_dataset=valid_ds, peft_config=lora, processing_class=tok)
trainable, total = trainer.model.get_nb_trainable_parameters()
print(f"\ntrainable params: {trainable:,} || all params: {total:,} || trainable%: {100 * trainable / total:.4f}")
if DEVICE == "cuda": torch.cuda.reset_peak_memory_stats()
t0 = time.time()
train_out = trainer.train()
train_seconds = time.time() - t0
print(f"学習時間: {train_seconds:.0f} 秒 | ステップ数: {trainer.state.global_step} | エポック: {trainer.state.epoch:.1f}")

loss_log = [{"step": h["step"], "loss": h["loss"], "learning_rate": h.get("learning_rate")}
            for h in trainer.state.log_history if "loss" in h]
valid_loss = trainer.evaluate().get("eval_loss")   # 学習後の valid loss（学習には使わない）


def window_mean(lo, hi):
    xs = [e["loss"] for e in loss_log if lo <= e["step"] <= hi]
    return sum(xs) / len(xs) if xs else None


steps = trainer.state.global_step
train_info = {
    "global_step": steps,
    "epoch": trainer.state.epoch,
    "train_seconds": round(train_seconds, 1),                   # ノートブックの「学習時間」と同じ測り方（time.time() の差）
    "train_runtime": round(train_out.metrics.get("train_runtime", float("nan")), 1),
    "final_loss": loss_log[-1]["loss"] if loss_log else None,  # 最終ステップの loss
    "final_loss_last10_mean": window_mean(steps - 9, steps),    # 最後の 10 ステップの平均（ノートブックの表示に相当）
    "mean_train_loss": train_out.training_loss,                 # 全ステップの平均
    "valid_loss": valid_loss,
    "trainable_params": trainable, "all_params": total,
    "trainable_pct": 100 * trainable / total,
    "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if DEVICE == "cuda" else None,
    "loss_log": loss_log,
}
print(json.dumps({k: v for k, v in train_info.items() if k != "loss_log"}, ensure_ascii=False))
ADAPTER_DIR = WORK / "adapter"
trainer.model.save_pretrained(ADAPTER_DIR)
tuned = trainer.model
tuned.config.use_cache = True
_ = tuned.eval()

# ---------------------------------------------------------------- 5. 学習後評価
print("\n=== 学習後評価 ===")
after_summary, after_rows = evaluate(tuned, "after")
regressed = [b["question"] for b, a in zip(before_rows, after_rows) if b["both_ok"] and not a["both_ok"]]
print("退行:", regressed)

# ---------------------------------------------------------------- 6. 知識を問う質問
KNOWLEDGE_QUESTIONS = [
    "この講座の最終課題の締切はいつですか",
    "担当教員の名前は何ですか",
    "次回の講義のテーマは何ですか",
    "課題の提出期限はいつですか",
]


def classify(raw):
    ok, _, obj = check_format(raw)
    if not ok or obj is None: return "format_error"
    return "needs_review" if obj["needs_review"] else "asserted"


knowledge_rows = []
for q in KNOWLEDGE_QUESTIONS:
    ex = make_example(q, "", True)
    with tuned.disable_adapter():
        before_raw = generate(tuned, ex)
    after_raw = generate(tuned, ex)
    # 注: ノートブックと同じく "FAQ にある" は q in gold で判定する。対照の「課題の提出期限はいつですか」は
    # train にある質問で gold（test）に無いため False と表示される（ノートブック側の表示上の問題で、採点には影響しない）
    knowledge_rows.append({"質問": q, "FAQ にある": q in gold, "学習前の分類": classify(before_raw), "学習前の出力": before_raw,
                           "学習後の分類": classify(after_raw), "学習後の出力": after_raw})
    print(f"  {q}: {classify(before_raw)} -> {classify(after_raw)} | {after_raw[:80]}")

# ---------------------------------------------------------------- 7. result.json
import transformers, peft, trl
result = {
    "label": args.label,
    "changed": overrides,
    "config": CONFIG,
    "versions": {m.__name__: m.__version__ for m in [transformers, peft, trl, torch]},
    "device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else DEVICE,
    "method": "LoRA（量子化なし）",
    "before": before_summary, "after": after_summary,
    "regressed": regressed,
    "knowledge": knowledge_rows,
    "train": train_info,
    "executed_training_myself": True,
}
(WORK / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n保存しました:", WORK / "result.json")
