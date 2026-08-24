import torch

from texelator.modeling import input_device
from texelator.runtime import TexelatorLinear


class OffloadedEmbedding:
    weight = torch.empty(1, device="meta")
    output_device = torch.device("cuda:0")


class TinyModel:
    def get_input_embeddings(self):
        return OffloadedEmbedding()


def test_input_device_uses_offload_destination():
    assert input_device(TinyModel()) == torch.device("cuda:0")


def test_texelator_linear_requires_handles():
    try:
        TexelatorLinear([])
    except ValueError as error:
        assert "at least one" in str(error)
    else:
        raise AssertionError("empty handles must be rejected")
