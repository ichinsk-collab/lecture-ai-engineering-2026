# 第3回 解答

## 演習

- 第1回（素のLLM）とエージェントの回答の違い：（Codespaces で実行して追記）
- `--trace` で見えたツール呼び出し（ツール名と引数）：（Codespaces で実行して追記）
- 1つの質問での LLM 呼び出し回数（`requests`）：（Codespaces で実行して追記）
- 追加したツールと、docstring に書いた説明：`make_quiz(lecture: int)`。docstring には「確認問題・クイズ・復習の問題を求められたときに使う」「答えは用語集の定義なので、問題文を自分で作らずこの結果を使って出題する」と、使う場面と結果の使い方を書いた。指定の回の用語が無い場合（`lecture=0` など）は、例外ではなく「問題を作れる回」の一覧を返す。`INSTRUCTIONS` にも「確認問題を求められたら make_quiz を使う」を足した。

## 宿題

- 取り組んだレベル：実装
- やったこと：
  - `app/tools.py` に `save_learning_log(title, content, lecture=None)` を追加した。`learning_log/` に `YYYYmmdd-HHMMSS_<題>.md` として、見出し・日時・関係する回・本文を Markdown で保存する。
    - 引数の説明は docstring の `Args:` 節に書いた。Pydantic AI はこれを JSON Schema の `description` に入れるので、`--list-tools` で各引数の説明がモデルに渡っていることを確認できる。
    - 取り消しにくい操作なので、ツール側でも防御した。既存ファイルは上書きしない（`open(..., "x")`、同名なら `-2` などを付ける）。題からファイル名を作るとき `/` や `.` を置き換え、保存先の外に出ないことを `resolve()` で確かめる。空の引数・長すぎる本文・`lecture < 1` は保存せず、理由をモデルに返す。
  - `app/agent.py` で `agent.tool_plain(tools.save_learning_log, requires_approval=True)` と登録し、`output_type=[str, DeferredToolRequests]` にした。
  - CLI に承認の流れを入れた（`run_with_approval`）。
    1. `run_sync(question)` の結果が `DeferredToolRequests` なら、`approvals` の呼び出しごとに、ツール名と引数（保存される題と本文）を表示して `[y/N]` で聞く。
    2. 結果を `DeferredToolResults(approvals={tool_call_id: True | ToolDenied(...)})` にまとめ、`message_history=result.all_messages()` と一緒に `run_sync` をもう一度呼ぶ。
    3. `usage=result.usage` を引き継ぎ、`[usage] requests` が承認の前後を合わせた LLM 呼び出し回数になるようにした。
  - 拒否側に倒す設計にした。`y`/`yes` 以外（空 Enter・`はい` など）と、入力できない環境（EOF）は拒否として扱う。拒否したときは `ToolDenied` に「保存をやり直さず、保存しなかったことを伝えること」というメッセージを入れてモデルに返し、`INSTRUCTIONS` にも同じことを書いた。それでもモデルが保存を呼び続ける場合に備え、承認のやり取りを最大3回に制限した。
  - `tests/test_learning_log.py` を追加した（API キー不要）。保存の中身、上書きしないこと、保存先の外に出ないこと、不正な引数、承認前は何も書かれないこと、承認で書かれること、拒否・EOF で書かれないこと、承認ループが上限で止まることを確かめる。
- 結果：
  - `uv run pytest -q` で 35 件すべて通った（既存の `tests/test_ask.py` の2件は `GEMINI_API_KEY` が無い環境では落ちるので、ダミーのキーを入れて実行した。Codespaces では secret があるので問題にならない）。
  - TestModel で CLI の流れを確かめた。`y` と答えると `learning_log/` にファイルができ、`[usage]` は `requests=2`（承認の前と後で1回ずつ）になった。入力が無いときは拒否になり、ファイルはできなかった。
  - 実際の Gemini での確認：（Codespaces で `uv run python -m app.agent --trace "LoRAとQLoRAの違いを調べて、学習ログに保存して"` を実行し、y と n の両方で結果を追記）
- 気づき・疑問：
  - `requires_approval=True` を付けると、ツールは実行されずに run が一度終わる。承認は「ツールの中で input() を呼ぶ」のではなく、run の外で人が判断して、会話の履歴と一緒に結果を渡し直す形になっている。そのため、CLI でも Web でも、承認する画面を差し替えるだけで同じエージェントを使える。
  - 承認のときに見るべきなのは、モデルが作った引数（何が保存されるか）である。そのため、承認画面には引数をそのまま全部表示した。
  - 承認は利用者への最後の防御であって、それだけに頼るのは危うい。ツール側でも、上書きしない・保存先の外に出ない・長さを制限する、という防御を入れた。
  - 疑問：拒否されたあとにモデルが同じ保存を呼び直すことはどの程度あるのか。実際の Gemini で、`ToolDenied` のメッセージの書き方によって再試行の頻度が変わるかを比べてみたい。
