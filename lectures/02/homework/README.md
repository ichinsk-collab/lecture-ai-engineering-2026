# 第2回 宿題（基礎）

`CONFIG` の `num_train_epochs` を 20 → 5 に変えて再学習し、元の設定と `format_rate`・`content_rate`・最終 loss・学習時間を並べて比べた。

| ファイル | 内容 |
| --- | --- |
| `homework_basic.ipynb` | Colab（T4）で上から実行するノートブック。下の 2 つのスクリプトを書き出して実行する |
| `run_lora.py` | `notebooks/lora_format_tuning.ipynb` のセクション 1〜7 と同じ処理を 1 プロセスで実行し、`result.json` を作る。1 回の実行で 1 つの設定 |
| `compare.py` | `result.json` を 2 つ並べて `comparison.md` と `answer.md` の下書きを作る |
| `results/` | 実行結果（`baseline/`・`epochs5/` の `result.json`・`eval_before.json`・`eval_after.json`、`comparison.md`） |

## ノートブックとの違い

学習・評価の処理（データ、プロンプト、採点、`SFTConfig` の学習に関わる設定）はノートブックと同じ。違いは記録だけ。

- `result.json` に `train`（最終 loss、最後の 10 ステップの loss 平均、全ステップの平均、学習後の valid loss、学習時間、trainable%、loss の推移）を追加した。宿題で記録する最終 loss と学習時間が、ノートブックの `result.json` には入っていないため。
- `logging_steps` を 10 → 1 にした。10 ステップしか学習しない設定では、10 ステップごとの記録だと最終 loss が「1〜10 ステップの平均」になり、元の設定の「31〜40 ステップの平均」と比べられないため。ログの間隔だけの設定で、勾配や学習率には影響しない。
- 設定ごとに別プロセスで実行する。学習後のモデルが残らないので、セッションを再起動して上から実行するのと同じ条件になる。

## 再現

```bash
python run_lora.py --label baseline
python run_lora.py --label epochs5 --num_train_epochs 5
python compare.py --baseline /content/qlora_lab/hw/baseline --variant /content/qlora_lab/hw/epochs5 --out /content/qlora_lab/hw
```
