"""第3回: エージェントに持たせるツール。

ここに書いた関数の docstring と型ヒントが、そのままモデルへの「ツールの説明文」になる。
説明文を書き換えると、モデルがツールを使うかどうか・どんな引数で呼ぶかが変わる。
"""

import json
import random
import re
from datetime import datetime
from functools import cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_RESULTS = 3

# 第3回 宿題（実装）: 学習ログの保存先。個人の記録なので .gitignore で共有しない
LOG_DIR = Path(__file__).resolve().parent.parent / "learning_log"
MAX_TITLE_CHARS = 100
MAX_LOG_CHARS = 10_000


@cache
def load_glossary() -> list[dict]:
    with open(DATA_DIR / "glossary.json", encoding="utf-8") as f:
        return json.load(f)["terms"]


def _normalize(text: str) -> str:
    return text.casefold().replace(" ", "").replace("　", "")


def _names(entry: dict) -> list[str]:
    return [_normalize(entry["term"]), _normalize(entry["id"])] + [
        _normalize(a) for a in entry["aliases"]
    ]


def search_glossary(query: str) -> str:
    """講座の用語集から用語を検索し、定義と、どの回で扱うかを返す。

    AIエンジニアリングの用語や、この講座（AIE2026）に固有の言葉について
    答える前に使うこと。query には調べたい用語を1つだけ入れる（例: "LoRA"）。
    """
    q = _normalize(query)
    if not q:
        return "検索語が空です。調べたい用語を1つ指定してください。"
    glossary = load_glossary()

    exact = [e for e in glossary if q in _names(e)]
    # 「r」のような短い別名が何にでも当たらないよう、名前側が短いものは除く
    partial = [
        e for e in glossary
        if e not in exact and any(q in n or (len(n) >= 3 and n in q) for n in _names(e))
    ]
    in_definition = [
        e for e in glossary
        if e not in exact and e not in partial and q in _normalize(e["definition"])
    ]
    hits = (exact + partial + in_definition)[:MAX_RESULTS]

    if not hits:
        return f"用語集に「{query}」は見つかりませんでした。"
    return json.dumps(
        [
            {
                "term": e["term"],
                "lecture": e["lecture"],
                "definition": e["definition"],
                "related": e["related"],
            }
            for e in hits
        ],
        ensure_ascii=False,
    )


def get_lecture_summary(lecture: int) -> str:
    """指定した回（1始まりの整数）の講義の要点を返す。

    「第3回では何を学ぶ？」のように、回の内容や関係を聞かれたときに使う。
    """
    path = DATA_DIR / "lectures" / f"{lecture:02d}.md"
    if not path.exists():
        available = sorted(p.stem for p in (DATA_DIR / "lectures").glob("*.md"))
        return f"第{lecture}回の要点はまだありません。用意されている回: {', '.join(available)}"
    return path.read_text(encoding="utf-8")


def make_quiz(lecture: int) -> str:
    """指定した回（1始まりの整数）の用語から確認問題を1問作り、問題と答えを返す。

    「第2回の確認問題を出して」「第3回の用語でクイズを出して」のように、
    確認問題・クイズ・復習の問題を求められたときに使う。
    答えは用語集の定義なので、問題文を自分で作らずこの結果を使って出題する。
    """
    candidates = [e for e in load_glossary() if e["lecture"] == lecture]
    if not candidates:
        available = sorted({e["lecture"] for e in load_glossary()})
        return (
            f"第{lecture}回の用語は用語集にありません。"
            f"問題を作れる回: {', '.join(f'第{n}回' for n in available)}"
        )
    entry = random.choice(candidates)
    return json.dumps(
        {"question": f"「{entry['term']}」の意味は？", "answer": entry["definition"]},
        ensure_ascii=False,
    )


def _slugify(title: str) -> str:
    """ファイル名に使える形にする。英数字・日本語・_・- 以外（/ や . など）は - に置き換える。"""
    slug = re.sub(r"[^\w\-]+", "-", title).strip("-_")
    return slug[:40] or "log"


def save_learning_log(title: str, content: str, lecture: int | None = None) -> str:
    """学習の記録を learning_log/ フォルダに Markdown ファイルとして保存する。

    利用者が「今日学んだことを記録して」「この説明をメモしておいて」のように、
    学習の記録を残すよう明示的に頼んだときだけ使う。頼まれていないのに使わない。
    実行の前に利用者の承認が入り、拒否されたら保存されない。

    Args:
        title: 記録の見出し。短く具体的に（例: "LoRAとQLoRAの違い"）。
        content: 記録する本文（Markdown）。利用者の発言と、この会話で確認した内容だけで書く。
        lecture: 関係する回（第N回、1始まりの整数）。分からなければ省略する。
    """
    title = title.strip()
    content = content.strip()
    if not title or not content:
        return "保存しませんでした: title と content の両方を指定してください。"
    if len(title) > MAX_TITLE_CHARS:
        return f"保存しませんでした: title は {MAX_TITLE_CHARS} 文字以内にしてください。"
    if len(content) > MAX_LOG_CHARS:
        return f"保存しませんでした: content は {MAX_LOG_CHARS} 文字以内にしてください。"
    if lecture is not None and lecture < 1:
        return "保存しませんでした: lecture は1以上の整数にするか、省略してください。"

    now = datetime.now().astimezone()
    meta = [f"- 日時: {now.isoformat(timespec='seconds')}"]
    if lecture is not None:
        meta.append(f"- 関係する回: 第{lecture}回")
    text = f"# {title}\n\n" + "\n".join(meta) + f"\n\n{content}\n"

    log_dir = LOG_DIR.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{now:%Y%m%d-%H%M%S}_{_slugify(title)}"
    for n in range(1, 100):
        path = log_dir / (f"{stem}.md" if n == 1 else f"{stem}-{n}.md")
        if path.resolve().parent != log_dir:  # 念のため：保存先の外には書かない
            return "保存しませんでした: ファイル名が不正です。"
        try:
            # "x" は既にファイルがあると失敗する。既存の記録を上書きしないため
            with open(path, "x", encoding="utf-8") as f:
                f.write(text)
        except FileExistsError:
            continue
        return f"学習の記録を保存しました: {LOG_DIR.name}/{path.name}"
    return "保存しませんでした: 同じ名前のファイルが多すぎます。title を変えてください。"
