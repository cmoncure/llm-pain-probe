import pytest
import torch

from llm_pain_probe.metrics import compute_probe_metrics


def test_projection_uses_unit_direction() -> None:
    hidden = torch.tensor([[3.0, 4.0]])
    direction = torch.tensor([2.0, 0.0])

    metrics = compute_probe_metrics(hidden, direction)

    assert metrics.projection.item() == pytest.approx(3.0)
    assert metrics.cosine.item() == pytest.approx(3.0 / 5.0)
    assert metrics.residual_rms.item() == pytest.approx((25.0 / 2.0) ** 0.5)
    assert metrics.control_z is None


def test_control_z_score() -> None:
    hidden = torch.tensor([[3.0, 4.0]])
    direction = torch.tensor([1.0, 0.0])

    metrics = compute_probe_metrics(
        hidden,
        direction,
        control_mean=1.0,
        control_std=0.5,
    )

    assert metrics.control_z is not None
    assert metrics.control_z.item() == pytest.approx(4.0)


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not match"):
        compute_probe_metrics(torch.ones(2, 3), torch.ones(4))
