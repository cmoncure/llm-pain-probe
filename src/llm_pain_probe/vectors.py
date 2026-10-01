from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ProbeVector:
    """A model- and layer-specific residual-stream direction."""

    values: Tensor
    model: str
    layer: int
    model_revision: str | None = None
    control_mean: float | None = None
    control_std: float | None = None
    label: str = "pain-axis"

    def __post_init__(self) -> None:
        if self.values.ndim != 1:
            raise ValueError("ProbeVector.values must be rank-1")
        if (self.control_mean is None) != (self.control_std is None):
            raise ValueError("control_mean and control_std must be provided together")
        if self.control_std is not None and self.control_std <= 0:
            raise ValueError("control_std must be positive")

    @property
    def hidden_dim(self) -> int:
        return int(self.values.shape[0])

    def unit(self) -> Tensor:
        v = self.values.float()
        norm = torch.linalg.vector_norm(v)
        if norm == 0:
            raise ValueError("cannot normalize a zero vector")
        return v / norm

    def save(self, path: str | Path) -> None:
        metadata = {
            "model": self.model,
            "layer": str(self.layer),
            "label": self.label,
        }
        if self.model_revision is not None:
            metadata["model_revision"] = self.model_revision
        if self.control_mean is not None:
            metadata["control_mean"] = repr(float(self.control_mean))
            metadata["control_std"] = repr(float(self.control_std))

        save_file(
            {"direction": self.values.detach().cpu().contiguous()},
            str(path),
            metadata=metadata,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ProbeVector":
        path = str(path)
        tensors = load_file(path)
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}

        return cls(
            values=tensors["direction"],
            model=metadata["model"],
            layer=int(metadata["layer"]),
            model_revision=metadata.get("model_revision"),
            control_mean=(
                float(metadata["control_mean"]) if "control_mean" in metadata else None
            ),
            control_std=(
                float(metadata["control_std"]) if "control_std" in metadata else None
            ),
            label=metadata.get("label", "pain-axis"),
        )
