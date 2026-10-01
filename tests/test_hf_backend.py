import torch
from torch import nn

from llm_pain_probe.backends.hf import forward_with_residual_capture


class AddOneBlock(nn.Module):
    def forward(self, hidden):
        return (hidden + 1.0,)


class TinyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([AddOneBlock(), AddOneBlock()])


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyCore()

    def forward(self, input_ids, attention_mask=None):
        hidden = input_ids.float().unsqueeze(-1)
        for block in self.model.layers:
            hidden = block(hidden)[0]
        return hidden


def test_capture_is_post_block_output() -> None:
    model = TinyCausalLM()
    input_ids = torch.tensor([[2, 4]])

    output, captures = forward_with_residual_capture(
        model,
        input_ids=input_ids,
        layers=[0, 1],
    )

    assert torch.equal(captures[0], torch.tensor([[[3.0], [5.0]]]))
    assert torch.equal(captures[1], torch.tensor([[[4.0], [6.0]]]))
    assert torch.equal(output, captures[1])
