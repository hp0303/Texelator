import pytest

from texelator.packager import artifact_to_text_name, source_to_text_key


def test_qwen38_full_to_text_mapping():
    full = "model.language_model.layers.63.mlp.down_proj"
    assert artifact_to_text_name(full) == "model.layers.63.mlp.down_proj"
    assert artifact_to_text_name("lm_head") == "lm_head"
    assert source_to_text_key(full + ".weight") == "model.layers.63.mlp.down_proj.weight"
    assert source_to_text_key("model.visual.patch_embed.weight") is None


def test_qwen38_mapping_rejects_unknown_modules():
    with pytest.raises(RuntimeError):
        artifact_to_text_name("model.visual.proj")
