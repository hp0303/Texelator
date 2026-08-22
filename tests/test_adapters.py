import torch.nn as nn

from texelator.adapters import discover_linears


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleDict({
            "q_proj": nn.Linear(32, 16, bias=False),
            "down_proj": nn.Linear(30, 16, bias=False),
        })])


def test_generic_discovery_and_shape_gate():
    specs = discover_linears(ToyModel())
    assert [item.op for item in specs] == ["down_proj", "q_proj"]
    assert specs[0].supported is False
    assert specs[1].supported is True

