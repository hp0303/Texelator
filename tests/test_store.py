from pathlib import Path

import pytest

from texelator.store import validate_local_model


def test_validate_local_model(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"test")
    assert validate_local_model(tmp_path) == tmp_path.resolve()


def test_reject_directory_without_weights(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(RuntimeError, match="checkpoint weights"):
        validate_local_model(tmp_path)

