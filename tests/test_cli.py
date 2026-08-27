from texelator.cli import _prompt_ids, _remaining_context_tokens, build_parser


def test_commands_are_separate():
    parser = build_parser()
    assert parser.parse_args(["model", "register", "/tmp/model", "--name", "local"]).model_command == "register"
    assert parser.parse_args(["ptq", "local"]).command == "ptq"
    assert parser.parse_args(["benchmark", "/tmp/artifact"]).command == "benchmark"
    prefill = parser.parse_args([
        "prefill-benchmark", "/tmp/artifact", "--tokens", "1024", "--runs", "5",
    ])
    assert prefill.command == "prefill-benchmark"
    assert prefill.tokens == 1024
    assert prefill.runs == 5
    assert parser.parse_args(["run", "/tmp/artifact"]).command == "run"
    assert parser.parse_args(["pull", "qwen3.8:27b"]).command == "pull"
    assert parser.parse_args(["download", "qwen3.8:27b"]).command == "download"
    assert parser.parse_args(["convert", "local"]).command == "convert"
    assert parser.parse_args(["tune", "/tmp/artifact"]).command == "tune"
    packaged = parser.parse_args([
        "package", "/tmp/artifact", "--source", "qwen38", "--output", "/tmp/output",
    ])
    assert packaged.command == "package"


def test_thinking_is_opt_in_and_response_has_no_default_cap():
    parser = build_parser()
    normal = parser.parse_args(["run", "artifact"])
    thinking = parser.parse_args(["run", "artifact", "--thinking"])
    assert normal.thinking is False
    assert thinking.thinking is True
    assert normal.max_new_tokens is None
    assert parser.parse_args(["run", "artifact", "--no-thinking"]).thinking is False
    assert parser.parse_args(["run", "artifact", "--max-new-tokens", "0"]).max_new_tokens == 0


def test_chat_template_receives_thinking_switch():
    class Tokenizer:
        chat_template = "template"

        def apply_chat_template(self, messages, **kwargs):
            self.kwargs = kwargs
            import torch
            return torch.tensor([[1, 2]])

    tokenizer = Tokenizer()
    _prompt_ids(tokenizer, [{"role": "user", "content": "hello"}], False, False)
    assert tokenizer.kwargs["enable_thinking"] is False


def test_unlimited_generation_uses_remaining_model_context():
    class Value:
        pass

    model = Value()
    model.config = Value()
    model.config.max_position_embeddings = 4096
    model.config.text_config = None
    tokenizer = Value()
    tokenizer.model_max_length = 8192
    assert _remaining_context_tokens(model, tokenizer, 512) == 3584
