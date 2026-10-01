import torch

from llm_pain_probe.vectors import ProbeVector


def test_vector_round_trip(tmp_path) -> None:
    path = tmp_path / "pain.safetensors"
    original = ProbeVector(
        values=torch.tensor([1.0, 2.0, 3.0]),
        model="example/model",
        model_revision="deadbeef",
        layer=7,
        control_mean=0.25,
        control_std=1.5,
    )

    original.save(path)
    loaded = ProbeVector.load(path)

    assert loaded.model == original.model
    assert loaded.model_revision == original.model_revision
    assert loaded.layer == original.layer
    assert loaded.control_mean == original.control_mean
    assert loaded.control_std == original.control_std
    assert torch.equal(loaded.values, original.values)
