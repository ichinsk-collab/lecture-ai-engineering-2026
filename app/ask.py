"""第1回: 素のLLMに会話履歴を保って質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールは使わず、会話履歴だけを保つ「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"
CHAT_SYSTEM_INSTRUCTION = """\
会話履歴を参照し、ユーザーが明示した目的、前提知識、希望する詳しさや回答形式、
これまでの質問・修正要望を踏まえて回答を調整してください。会話が進むにつれて、
ユーザーが示した理解度や希望に合う説明にしてください。
好みが明確でない場合は決めつけず、現在の質問と明示された指示を優先してください。
個人情報やセンシティブな属性を推測せず、事実の正確さを優先してください。
"""


def create_chat(client):
    config = genai.types.GenerateContentConfig(
        system_instruction=CHAT_SYSTEM_INSTRUCTION
    )
    return client.chats.create(model=MODEL, config=config)


def ask(question: str, chat=None) -> str:
    if chat is None:
        client = genai.Client(api_key=get_api_key())
        chat = create_chat(client)
    response = chat.send_message(question)

    usage = response.usage_metadata
    input_tokens = getattr(usage, "prompt_token_count", None)
    output_tokens = getattr(usage, "candidates_token_count", None)
    print(
        f"[usage] 入力: {input_tokens if input_tokens is not None else '不明'} トークン / "
        f"出力: {output_tokens if output_tokens is not None else '不明'} トークン"
    )

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        client = genai.Client(api_key=get_api_key())
        chat = create_chat(client)
        question = args.question
        while question:
            print(f"[質問] {question}\n")
            print(ask(question, chat))
            try:
                question = input("\n続けて質問してください（終了するには空欄）: ").strip()
            except EOFError:
                break
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
