from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .store import ModelRecord


def load_model(record: ModelRecord, device_map: str = "cuda"):
    kwargs = {
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
    }
    if device_map == "cuda":
        model = AutoModelForCausalLM.from_pretrained(record.source, **kwargs).cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(record.source, device_map=device_map, **kwargs)
    return model.eval()


def load_tokenizer(record: ModelRecord):
    return AutoTokenizer.from_pretrained(record.source, local_files_only=True)


def input_device(model) -> torch.device:
    device = model.get_input_embeddings().weight.device
    if device.type == "meta":
        raise RuntimeError("input embedding remains on a meta device")
    return device

