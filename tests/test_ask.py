from types import SimpleNamespace

from app import ask as ask_module


def test_ask_prints_input_and_output_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(
        text="answer",
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=34,
        ),
    )
    fake_chat = SimpleNamespace(send_message=lambda question: response)
    fake_client = SimpleNamespace(chats=SimpleNamespace(create=lambda **kwargs: fake_chat))
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: fake_client)

    assert ask_module.ask("question") == "answer"
    assert "[usage] 入力: 12 トークン / 出力: 34 トークン" in capsys.readouterr().out


def test_ask_handles_missing_usage_metadata(monkeypatch, capsys):
    response = SimpleNamespace(text="answer", usage_metadata=None)
    fake_chat = SimpleNamespace(send_message=lambda question: response)
    fake_client = SimpleNamespace(chats=SimpleNamespace(create=lambda **kwargs: fake_chat))
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: fake_client)

    assert ask_module.ask("question") == "answer"
    assert "[usage] 入力: 不明 トークン / 出力: 不明 トークン" in capsys.readouterr().out


def test_main_reuses_chat_for_follow_up_questions(monkeypatch, capsys):
    received_questions = []
    chat_configs = []
    response = SimpleNamespace(text="answer", usage_metadata=None)
    fake_chat = SimpleNamespace(
        send_message=lambda question: (
            received_questions.append(question) or response
        )
    )
    fake_client = SimpleNamespace(
        chats=SimpleNamespace(
            create=lambda **kwargs: chat_configs.append(kwargs["config"]) or fake_chat
        )
    )
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: fake_client)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr("sys.argv", ["app.ask", "最初の質問"])
    questions = iter(["続きの質問", ""])
    monkeypatch.setattr("builtins.input", lambda prompt: next(questions))

    ask_module.main()

    assert received_questions == ["最初の質問", "続きの質問"]
    assert len(chat_configs) == 1
    assert "会話履歴" in chat_configs[0].system_instruction
    assert "センシティブな属性を推測せず" in chat_configs[0].system_instruction
    assert capsys.readouterr().out.count("answer") == 2