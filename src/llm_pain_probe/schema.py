from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

ProbePhase = Literal["prefill", "decode"]


@dataclass(frozen=True, slots=True)
class ProbeEvent:
    """One scalar probe observation for one token at one layer."""

    run_id: str
    model: str
    model_revision: str | None
    backend: str
    backend_revision: str | None
    phase: ProbePhase
    token_index: int
    token_id: int
    layer: int
    projection: float
    cosine: float
    residual_rms: float
    control_z: float | None = None
    request_id: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
