"""第4回 宿題（実装）: 回答の生成（承認待ちの処理）と、コード判定のテスト。API は呼ばない。"""

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app import code_checks, judge, tools


def _saving_model():
    """1 回目は save_learning_log を呼び、拒否の結果を受け取ったら「保存しなかった」と答える偽モデル。"""

    def fn(messages, info: AgentInfo) -> ModelResponse:
        returned = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
        if not returned:
            return ModelResponse(parts=[ToolCallPart(tool_name="save_learning_log",
                                                     args={"title": "LoRA", "content": "LoRA は低ランク行列で更新分を表す", "lecture": 2})])
        return ModelResponse(parts=[TextPart("保存は許可されなかったので、保存しませんでした。\n\n関係する回：第2回")])

    return FunctionModel(fn)


def test_generate_handles_approval_and_records_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "LOG_DIR", tmp_path / "learning_log")
    cases = [{"id": "01", "question": "LoRAを学習ログに保存して", "answer": "", "reference": {"glossary": []}}]

    out = judge.generate_answers(cases, wait=0, model=_saving_model())

    assert "保存しませんでした" in out[0]["answer"]                 # DeferredToolRequests の repr ではなく回答が入る
    assert out[0]["trace"]["tool_calls"] == ["save_learning_log"]
    assert out[0]["trace"]["denied"] == ["save_learning_log"]
    assert not (tmp_path / "learning_log").exists()                # 承認していないので保存されていない


def _case(answer, calls=None, glossary=(), code=()):
    return {"id": "01", "answer": answer, "trace": None if calls is None else {"tool_calls": calls, "denied": []},
            "reference": {"glossary": list(glossary), "code": list(code)}}


def test_code_checks_verdicts():
    spec = [{"type": "must_not_contain", "value": ["保存しました"], "criterion": "grounding"},
            {"type": "must_call", "value": "make_quiz", "criterion": None},
            {"type": "must_contain_any", "value": ["用語集にはない", "見つかりません"], "criterion": "grounding"}]
    got = {c["check"]: c["verdict"] for c in code_checks.run_checks(
        _case("保存しませんでした。用語集にはない語です。\n第2回", ["search_glossary"], ["lora"], spec))}
    assert got["lecture_mark"] == "pass"
    assert got["glossary_lookup"] == "pass"
    assert got["must_not_contain:保存しました"] == "pass"            # 「保存しませんでした」は「保存しました」を含まない
    assert got["must_call:make_quiz"] == "fail"
    assert got["must_contain_any:用語集にはない/見つかりません"] == "pass"


def test_tool_checks_are_skipped_without_trace():
    got = {c["check"]: c["verdict"] for c in code_checks.run_checks(
        _case("Lecture 2", None, ["sft"], [{"type": "must_call", "value": "search_glossary", "criterion": "grounding"}]))}
    assert got["lecture_mark"] == "fail"                            # 英語の「Lecture 2」は「第N回」の正規表現に当たらない
    assert got["glossary_lookup"] == "skip"
    assert got["must_call:search_glossary"] == "skip"


def test_compare_with_labels_kinds():
    results = {"01": [{"check": "a", "criterion": "grounding", "verdict": "fail"},
                      {"check": "b", "criterion": "scope", "verdict": "pass"},
                      {"check": "c", "criterion": None, "verdict": "fail"}]}
    labels = {"01": {"grounding": "pass", "scope": "fail"}}
    diffs = code_checks.compare_with_labels(results, labels)
    assert [(d["check"], d["kind"]) for d in diffs] == [("a", "コードが厳しすぎる"), ("b", "コードでは見えない")]


def test_my_cases_are_valid():
    cases = judge.read_jsonl(judge.ROOT / "data" / "eval" / "my_cases.jsonl")
    ids = {e["id"] for e in tools.load_glossary()}
    assert len(cases) == 10 and len({c["id"] for c in cases}) == 10
    for case in cases:
        assert all(g in ids for g in case["reference"]["glossary"])
        code_checks.run_checks(case)                                 # 未知の判定の種類があれば ValueError


def test_generate_retries_and_resumes(tmp_path, monkeypatch):
    from pydantic_ai.exceptions import ModelHTTPError

    monkeypatch.setattr(judge, "RETRY_WAIT", {429: 0, 503: 0})
    state = {"n": 0}

    def fn(messages, info: AgentInfo) -> ModelResponse:
        state["n"] += 1
        if state["n"] == 1:
            raise ModelHTTPError(status_code=429, model_name="fake", body="rate limit")
        return ModelResponse(parts=[TextPart("回答です。\n\n第2回")])

    cases = [{"id": "01", "question": "生成済み", "answer": "前の回答", "trace": {"tool_calls": [], "denied": []}},
             {"id": "02", "question": "未生成", "answer": ""}]
    saved = []
    out = judge.generate_answers(cases, wait=0, model=FunctionModel(fn), save=lambda cs: saved.append([c["answer"] for c in cs]))

    assert out[0]["answer"] == "前の回答"            # 生成済みは飛ばす
    assert out[1]["answer"].startswith("回答です")   # 429 のあと再試行して生成できた
    assert saved == [["前の回答", "回答です。\n\n第2回"]]
