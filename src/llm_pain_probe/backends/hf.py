from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn


def resolve_decoder_layers(model: nn.Module) -> list[nn.Module]:
    """Resolve decoder blocks for common Hugging Face causal-LM layouts."""
    candidates = (
        ("model", "layers"),
        ("model", "model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("model", "gpt_neox", "layers"),
    )

    for path in candidates:
        obj: Any = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, (nn.ModuleList, list, tuple)) and len(obj) > 0:
            return list(obj)

    raise ValueError(
        "Could not resolve decoder layers. Add the model's module path to "
        "resolve_decoder_layers() explicitly."
    )


def _first_tensor(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0]
    raise TypeError(f"Unsupported decoder-block output type: {type(output)!r}")


@contextmanager
def capture_post_block_residuals(
    model: nn.Module,
    layers: Iterable[int],
):
    """Capture selected decoder-block outputs without copying them to the host."""
    blocks = resolve_decoder_layers(model)
    requested = sorted(set(int(layer) for layer in layers))
    if not requested:
        raise ValueError("at least one layer must be requested")

    captures: dict[int, Tensor] = {}
    handles: list[Any] = []

    for layer in requested:
        if layer < 0 or layer >= len(blocks):
            raise IndexError(f"layer {layer} is outside [0, {len(blocks) - 1}]")

        def hook(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
            *,
            idx: int = layer,
        ) -> None:
            captures[idx] = _first_tensor(output).detach()

        handles.append(blocks[layer].register_forward_hook(hook))

    try:
        yield captures
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def forward_with_residual_capture(
    model: nn.Module,
    *,
    input_ids: Tensor,
    layers: Iterable[int],
    attention_mask: Tensor | None = None,
    **forward_kwargs: Any,
) -> tuple[Any, dict[int, Tensor]]:
    """Run one forward pass and return outputs plus selected post-block states."""
    with capture_post_block_residuals(model, layers) as captures:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **forward_kwargs,
        )
    return outputs, captures
