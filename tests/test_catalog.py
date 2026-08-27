from texelator import catalog


def test_catalog_selects_gpu_specific_repo(monkeypatch):
    monkeypatch.delenv("TEXELATOR_MODEL_REPO_QWEN3_8_27B", raising=False)
    assert catalog.catalog_repo("qwen3.8:27b", "8.9") == "hp0303/Qwen3.8-27B-Texelator-AWBC4"
    assert catalog.catalog_repo("qwen3.8:27b", "12.0") == "hp0303/Qwen3.8-27B-Texelator-AWBC4-BF16Cal"


def test_catalog_repo_override_has_shell_safe_name(monkeypatch):
    monkeypatch.setenv("TEXELATOR_MODEL_REPO_QWEN3_8_27B", "local/private")
    assert catalog.catalog_repo("qwen3.8:27b", "8.9") == "local/private"
