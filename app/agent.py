"""第3回: Pydantic AI で作る学習支援エージェント。

使い方:
    uv run python -m app.agent "AIE2026の講座で使う「lecNN-start タグ」とは何ですか？"
    uv run python -m app.agent --trace "LoRAとQLoRAの違いは？"   # ツール呼び出しを表示
    uv run python -m app.agent "LoRAとは何かを調べて、学習ログに保存して"  # 第3回 宿題: 保存の前に承認を求める
    uv run python -m app.agent --mcp --trace "LoRAとは？"        # 第5回: ツールを MCP サーバーから受け取る
    uv run python -m app.agent --mcp --list-tools               # 第5回: モデルに見えるツール一覧（APIキー不要）

エージェント = モデル + ループ + ツール。
モデルは tools.py の関数を「ツール」として呼び出し、その結果を見て回答を作る。
第5回では、同じツールを MCP サーバー（app/mcp_server.py）経由で受け取る。
"""

import argparse
import inspect
import json
from pathlib import Path

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, ToolDenied
from pydantic_ai.exceptions import ModelHTTPError
from fastmcp import Client
from pydantic_ai.mcp import MCPToolset, StdioTransport
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.google import GoogleProvider

from app import tools
from app.config import MODEL, api_error_message, get_api_key

INSTRUCTIONS = """\
あなたはAIエンジニアリング講座（AIE2026）の学習支援エージェントです。
- 用語や講座に固有のことを答える前に、必ず search_glossary で用語集を確認する。
- 回の内容を聞かれたら get_lecture_summary を使う。
- 用語集に見つからないことは、推測で断定せず「用語集にはない」と伝える。
- 確認問題を求められたら make_quiz を使い、その問題を出題する。答えは問題のあとに「答え」として分けて示す。
- 学習の記録を残すよう頼まれたときだけ save_learning_log を使う。本文は会話で確認した内容だけで書く。
- save_learning_log が拒否されたら、同じ保存をやり直さず、保存しなかったことを伝える。
- 回答の最後に、関係する回（第N回）を示す。
- 宿題の答えそのものは書かず、考え方のヒントで導く。
"""

# モデルは実行時に渡す（テストではAPIを呼ばない TestModel に差し替えるため）
# 第3回 宿題: 承認待ちのツールを呼ぶと、回答（str）の代わりに DeferredToolRequests が返る
agent = Agent(instructions=INSTRUCTIONS, output_type=[str, DeferredToolRequests])
agent.tool_plain(tools.search_glossary)
agent.tool_plain(tools.get_lecture_summary)
agent.tool_plain(tools.make_quiz)
# ファイルへの保存は取り消しにくいので、実行前に利用者の承認を挟む（Human-in-the-Loop）
agent.tool_plain(tools.save_learning_log, requires_approval=True)

# 承認のやり取りの上限。モデルが拒否後も保存を呼び続けても、ループが終わるようにする
MAX_APPROVAL_ROUNDS = 3
DENIED_MESSAGE = "利用者が保存を許可しなかった。保存をやり直さず、保存しなかったことを利用者に伝えること。"

# 第5回: 使う MCP サーバーの設定ファイル。Claude Desktop や Cursor と同じ "mcpServers" 形式
MCP_CONFIG = Path(__file__).resolve().parent / "mcp_servers.json"


def load_mcp_servers(config_path: Path = MCP_CONFIG) -> list[MCPToolset]:
    """設定ファイル（mcpServers 形式）を読み、サーバーごとに接続（MCPToolset）を作る。

    - "command" があれば stdio：サーバーをサブプロセスとして起動する
    - "url" があれば Streamable HTTP：起動済みのサーバーに接続する
    ツール名には「サーバー名_」を付ける（例: glossary_search_glossary）。複数のサーバーに
    同じ名前のツールがあっても区別できるようにするため。

    Pydantic AI には同じことをする load_mcp_toolsets() もある。ここで自分で組み立てているのは、
    古い SDK で作られたサーバー（fetch など）につなぐとき、接続方式の自動判定（"auto"）が
    警告を大量に出すため。"legacy" に固定すると新旧どちらのサーバーにも静かにつながる。
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))["mcpServers"]
    toolsets = []
    for name, spec in config.items():
        if "url" in spec:
            transport = spec["url"]
        else:
            transport = StdioTransport(spec["command"], spec.get("args", []), env=spec.get("env"))
        toolsets.append(MCPToolset(Client(transport, mode="legacy")).prefixed(name))
    return toolsets


def build_mcp_agent(config_path: Path = MCP_CONFIG) -> Agent:
    """第5回: 関数を直接登録する代わりに、MCP サーバーが公開するツールを使うエージェント。INSTRUCTIONS は同じ。"""
    return Agent(instructions=INSTRUCTIONS, toolsets=load_mcp_servers(config_path))


def build_model() -> GoogleModel:
    return GoogleModel(MODEL, provider=GoogleProvider(api_key=get_api_key()))


def print_tools(target: Agent) -> None:
    """モデルに見えているツールの一覧（名前・説明文・引数）を表示する。LLM は呼ばない。

    TestModel（テストで使う偽モデル）にツール一覧を渡し、そこから読み出す。
    MCP サーバーのツールも、第3回の関数ツールも、モデルから見れば同じ形をしている。
    """
    model = TestModel(call_tools=[])
    target.run_sync("ツール一覧", model=model)
    tools_seen = model.last_model_request_parameters.function_tools
    print(f"モデルに見えているツール: {len(tools_seen)} 個")
    for tool in tools_seen:
        print(f"\n[{tool.name}]")
        print("  " + inspect.cleandoc(tool.description or "").replace("\n", "\n  "))
        params = tool.parameters_json_schema.get("properties", {})
        print(f"  引数: {json.dumps(params, ensure_ascii=False)}")


def print_trace(result) -> None:
    """どのツールが、どの引数で呼ばれ、何を返したかを表示する。"""
    print("\n--- trace ---")
    for message in result.all_messages():
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                print(f"[tool call]   {part.tool_name}({part.args})")
            elif isinstance(part, ToolReturnPart):
                print(f"[tool return] {part.tool_name} -> {str(part.content)[:120]}…")
    print(f"[usage] {result.usage}")


def ask_approval(call: ToolCallPart, input_fn=input) -> bool | ToolDenied:
    """承認待ちのツール呼び出し1つについて、実行してよいかを利用者に聞く。

    y / yes のときだけ承認する。空 Enter・それ以外の入力・入力できない環境（EOF）は拒否として扱う
    （取り消しにくい操作なので、迷ったら実行しない側に倒す）。
    """
    print(f"\n[approval] {call.tool_name} を実行しようとしています")
    for key, value in call.args_as_dict().items():
        text = str(value)
        print(f"  {key}: " + text.replace("\n", "\n    "))
    try:
        answer = input_fn("実行してよいですか？ [y/N]: ")
    except EOFError:
        answer = ""
    if answer.strip().lower() in ("y", "yes"):
        return True
    return ToolDenied(DENIED_MESSAGE)


def resolve_approvals(requests: DeferredToolRequests, input_fn=input) -> DeferredToolResults:
    """DeferredToolRequests の承認待ちを1件ずつ利用者に聞き、DeferredToolResults にまとめる。"""
    if requests.calls:
        names = ", ".join(c.tool_name for c in requests.calls)
        raise RuntimeError(f"外部で実行するツール（{names}）には対応していません")
    return DeferredToolResults(
        approvals={call.tool_call_id: ask_approval(call, input_fn) for call in requests.approvals}
    )


def run_with_approval(target: Agent, question: str, model, input_fn=input,
                      max_rounds: int = MAX_APPROVAL_ROUNDS):
    """エージェントを実行し、承認待ちで止まったら利用者に聞いて、会話を引き継いで再開する。

    1回目: run_sync(question) → output が DeferredToolRequests なら承認待ち
    2回目以降: run_sync(message_history=それまでの会話, deferred_tool_results=承認の結果)
    usage を引き継いで、[usage] が全体の LLM 呼び出し回数になるようにする。
    """
    result = target.run_sync(question, model=model)
    for _ in range(max_rounds):
        if not isinstance(result.output, DeferredToolRequests):
            return result
        results = resolve_approvals(result.output, input_fn)
        result = target.run_sync(
            message_history=result.all_messages(),
            deferred_tool_results=results,
            model=model,
            usage=result.usage,
        )
    if isinstance(result.output, DeferredToolRequests):
        # 上限に達したら、残りは聞かずに全部拒否して、回答だけ作らせる
        results = DeferredToolResults(
            approvals={c.tool_call_id: ToolDenied(DENIED_MESSAGE) for c in result.output.approvals}
        )
        result = target.run_sync(
            message_history=result.all_messages(),
            deferred_tool_results=results,
            model=model,
            usage=result.usage,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="学習支援エージェントに質問する")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--trace", action="store_true", help="ツール呼び出しを表示する")
    parser.add_argument("--mcp", action="store_true", help="第5回: ツールを MCP サーバー（app/mcp_servers.json）から受け取る")
    parser.add_argument("--list-tools", action="store_true", help="モデルに見えるツール一覧を表示して終了する（APIキー不要）")
    args = parser.parse_args()

    target = build_mcp_agent() if args.mcp else agent
    if args.list_tools:
        print_tools(target)
        return
    if not args.question:
        parser.error("質問を指定してください")

    print(f"[model] {MODEL}")
    if args.mcp:
        print("[tools] MCP サーバー（app/mcp_servers.json）")
    try:
        result = run_with_approval(target, args.question, build_model())
    except ModelHTTPError as e:
        print(api_error_message(e.status_code, str(e.body)[:200]))
        raise SystemExit(1)
    if isinstance(result.output, DeferredToolRequests):
        print("承認の上限回数に達したため、回答を作らずに終了しました。")
    else:
        print(result.output)
    if args.trace:
        print_trace(result)


if __name__ == "__main__":
    main()
