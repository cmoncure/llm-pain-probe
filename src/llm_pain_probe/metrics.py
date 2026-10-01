from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ProbeMetrics:
    projection: Tensor
    cosine: Tensor
    residual_rms: Tensor
    control_z: Tensor | None


def compute_probe_metrics(
    hidden: Tensor,
    direction: Tensor,
    *,
    control_mean: float | Tensor | None = None,
    control_std: float | Tensor | None = None,
    eps: float = 1e-12,
) -> ProbeMetrics:
    """Project residual states onto a direction using float32 accumulation."""
    if hidden.ndim < 1:
        raise ValueError("hidden must have at least one dimension")
    if direction.ndim != 1:
        raise ValueError("direction must be rank-1")
    if hidden.shape[-1] != direction.shape[0]:
        raise ValueError(
            f"hidden dim {hidden.shape[-1]} does not match direction dim {direction.shape[0]}"
        )
    if (control_mean is None) != (control_std is None):
        raise ValueError("control_mean and control_std must be provided together")

    h = hidden.float()
    v = direction.to(device=h.device, dtype=torch.float32)
    v_norm = torch.linalg.vector_norm(v).clamp_min(eps)
    v_unit = v / v_norm

    projection = torch.einsum("...d,d->...", h, v_unit)
    h_norm = torch.linalg.vector_norm(h, dim=-1).clamp_min(eps)
    cosine = projection / h_norm
    residual_rms = torch.sqrt(torch.mean(h * h, dim=-1))

    control_z = None
    if control_mean is not None and control_std is not None:
        mean = torch.as_tensor(control_mean, dtype=torch.float32, device=h.device)
        std = torch.as_tensor(control_std, dtype=torch.float32, device=h.device)
        if torch.any(std <= 0):
            raise ValueError("control_std must be positive")
        control_z = (projection - mean) / std

    return ProbeMetrics(
        projection=projection,
        cosine=cosine,
        residual_rms=residual_rms,
        control_z=control_z,
    )
