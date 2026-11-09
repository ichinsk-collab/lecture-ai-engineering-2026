"""第3回 宿題（実装）: make_quiz と、承認付きの save_learning_log のテスト（LLMもAPIキーも使わない）。

保存先は tmp_path に差し替えるので、実際の learning_log/ には書かない。
"""

import json

import pytest
from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from app import agent as agent_module
from app import tools
from app.agent import agent, run_with_approval


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    path = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LOG_DIR", path)
    return path


def _returns(result, tool_name):
    return [
        p for m in result.all_messages() for p in m.parts
        if isinstance(p, ToolReturnPart) and p.tool_name == tool_name
    ]


def _save_only_model():
    # save_learning_log だけを呼ぶ偽モデル。引数はダミー（title="a" など）
    return TestModel(call_tools=["save_learning_log"])


# --- make_quiz ---------------------------------------------------------------

def test_make_quiz_uses_term_of_the_lecture():
    quiz = json.loads(tools.make_quiz(2))
    lecture2 = {e["term"]: e["definition"] for e in tools.load_glossary() if e["lecture"] == 2}
    term = quiz["question"].removeprefix("「").removesuffix("」の意味は？")
    assert term in lecture2
    assert quiz["answer"] == lecture2[term]


def test_make_quiz_reports_missing_lecture():
    message = tools.make_quiz(0)
    assert "ありません" in message and "第2回" in message


# --- save_learning_log（ツール単体） -------------------------------------------

def test_save_writes_markdown_into_log_dir(log_dir):
    message = tools.save_learning_log("LoRAとQLoRAの違い", "QLoRAは量子化したモデルにLoRAを付ける。", 2)
    files = list(log_dir.glob("*.md"))
    assert len(files) == 1 and files[0].name in message
    text = files[0].read_text(encoding="utf-8")
    assert text.startswith("# LoRAとQLoRAの違い\n")
    assert "第2回" in text and "量子化" in text


def test_save_does_not_overwrite_existing_log(log_dir):
    tools.save_learning_log("同じ題", "1つめ")
    tools.save_learning_log("同じ題", "2つめ")  # 同じ秒に同じ題でも、別のファイルになる
    texts = sorted(p.read_text(encoding="utf-8") for p in log_dir.glob("*.md"))
    assert len(texts) == 2 and "1つめ" in texts[0] and "2つめ" in texts[1]


def test_save_stays_inside_log_dir(log_dir):
    tools.save_learning_log("../../etc/passwd", "path traversal")
    files = list(log_dir.rglob("*"))
    assert len(files) == 1 and files[0].parent == log_dir
    assert not (log_dir.parent / "etc").exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"title": " ", "content": "本文"},
        {"title": "題", "content": ""},
        {"title": "x" * (tools.MAX_TITLE_CHARS + 1), "content": "本文"},
        {"title": "題", "content": "x" * (tools.MAX_LOG_CHARS + 1)},
        {"title": "題", "content": "本文", "lecture": 0},
    ],
)
def test_save_rejects_invalid_arguments(log_dir, kwargs):
    assert tools.save_learning_log(**kwargs).startswith("保存しませんでした")
    assert not log_dir.exists() or not any(log_dir.iterdir())


# --- save_learning_log（承認の流れ） --------------------------------------------

def test_save_waits_for_approval_and_does_not_write(log_dir):
    with agent.override(model=_save_only_model()):
        result = agent.run_sync("学習ログに保存して")
    assert isinstance(result.output, DeferredToolRequests)
    assert [c.tool_name for c in result.output.approvals] == ["save_learning_log"]
    assert not log_dir.exists()  # 承認前なので、まだ何も書かれていない


def test_approved_save_writes_file(log_dir, capsys):
    result = run_with_approval(agent, "学習ログに保存して", _save_only_model(), input_fn=lambda _: "y")
    assert isinstance(result.output, str)
    assert len(list(log_dir.glob("*.md"))) == 1
    assert "保存しました" in str(_returns(result, "save_learning_log")[0].content)
    assert "[approval] save_learning_log" in capsys.readouterr().out  # 利用者に引数を見せている
    assert result.usage.requests == 2  # 承認の前と後で1回ずつ。usage が引き継がれている


@pytest.mark.parametrize("answer", ["n", "", "はい"])
def test_denied_save_writes_nothing_and_tells_the_model(log_dir, answer):
    result = run_with_approval(agent, "学習ログに保存して", _save_only_model(), input_fn=lambda _: answer)
    assert isinstance(result.output, str)
    assert not log_dir.exists()
    assert str(_returns(result, "save_learning_log")[0].content) == agent_module.DENIED_MESSAGE


def test_eof_on_input_is_treated_as_denial(log_dir):
    def no_input(_):
        raise EOFError

    run_with_approval(agent, "学習ログに保存して", _save_only_model(), input_fn=no_input)
    assert not log_dir.exists()


def test_approval_loop_is_bounded(log_dir):
    """拒否されても保存を呼び続けるモデルでも、上限回数で打ち切って終わる。"""
    def always_save(messages, info):
        # 毎回 save_learning_log を呼ぶ偽モデル（拒否されても諦めない悪い例）
        return ModelResponse(parts=[ToolCallPart("save_learning_log", {"title": "t", "content": "c"})])

    asked = []

    def deny(prompt):
        asked.append(prompt)
        return "n"

    result = run_with_approval(agent, "保存して", FunctionModel(always_save), input_fn=deny, max_rounds=2)
    assert len(asked) == 2  # 上限までは聞くが、それ以上は聞かない
    assert isinstance(result.output, DeferredToolRequests)  # main() はこれを見て打ち切りを伝える
    assert not log_dir.exists()
