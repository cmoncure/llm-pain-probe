from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_pain_probe.backends.hf import forward_with_residual_capture
from llm_pain_probe.metrics import compute_probe_metrics
from llm_pain_probe.vectors import ProbeVector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a one-prompt HF residual probe.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--vector", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--last-n", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vector = ProbeVector.load(args.vector)

    if vector.model != args.model:
        raise ValueError(
            f"vector was built for {vector.model!r}, but --model is {args.model!r}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype="auto",
        device_map="auto",
    )
    encoded = tokenizer(args.prompt, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    _, captures = forward_with_residual_capture(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        layers=[vector.layer],
        use_cache=False,
    )

    hidden = captures[vector.layer][0]
    metrics = compute_probe_metrics(
        hidden,
        vector.values,
        control_mean=vector.control_mean,
        control_std=vector.control_std,
    )

    start = max(0, input_ids.shape[1] - args.last_n)
    for token_index in range(start, input_ids.shape[1]):
        token_id = int(input_ids[0, token_index])
        row = {
            "token_index": token_index,
            "token_id": token_id,
            "token_text": tokenizer.decode([token_id]),
            "layer": vector.layer,
            "projection": float(metrics.projection[token_index]),
            "cosine": float(metrics.cosine[token_index]),
            "residual_rms": float(metrics.residual_rms[token_index]),
            "control_z": (
                float(metrics.control_z[token_index])
                if metrics.control_z is not None
                else None
            ),
        }
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
