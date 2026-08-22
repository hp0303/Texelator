from texelator.cli import build_parser


def test_commands_are_separate():
    parser = build_parser()
    assert parser.parse_args(["model", "register", "/tmp/model", "--name", "local"]).model_command == "register"
    assert parser.parse_args(["convert", "local"]).command == "convert"
    assert parser.parse_args(["tune", "/tmp/artifact"]).command == "tune"
    assert parser.parse_args(["run", "/tmp/artifact"]).command == "run"

