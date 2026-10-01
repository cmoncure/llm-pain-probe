from __future__ import annotations

import argparse
from pathlib import Path

import torch

from llm_pain_probe.vectors import ProbeVector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Pain-axis pain_vectors.pt file to our safetensors artifact."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--key", default="s2_pain_vector")
    parser.add_argument("--layer", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.input, map_location="cpu", weights_only=True)

    if args.key not in payload:
        raise KeyError(f"{args.key!r} not found; available keys: {sorted(payload)}")

    layer = args.layer if args.layer is not None else int(payload["layer"])
    vector = ProbeVector(
        values=payload[args.key].detach().float().cpu(),
        model=args.model,
        model_revision=args.model_revision,
        layer=layer,
        label=args.key,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    vector.save(args.output)
    print(
        f"wrote {args.output} model={vector.model} layer={vector.layer} "
        f"hidden_dim={vector.hidden_dim} label={vector.label}"
    )


if __name__ == "__main__":
    main()
