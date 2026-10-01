# llm-pain-probe

Experimental instrumentation for measuring residual-stream projections onto the “pain axis” described by Valen Research.

The first milestone is backend parity: for the same model, token sequence, layer, and vector, a Hugging Face reference implementation and a vLLM implementation should report matching scalar projections within expected numerical tolerance.

## Scope

Phase 1 focuses on:

- a backend-independent probe/event schema;
- a Hugging Face reference probe;
- vector artifact metadata and normalization;
- tests for projection, cosine, RMS, and control-normalized z-scores;
- a later vLLM adapter using the same semantics.

Large-model llama.cpp instrumentation will follow only after HF↔vLLM parity is established.

## Scientific convention

For token position \(t\) and decoder block \(l\), the primary observable is

\[
P(l,t) = h_{l,t}^{T}\hat v_l
\]

where \(h_{l,t}\) is the post-block residual stream and \(\hat v_l\) is a unit direction for that layer.

We additionally record cosine alignment, residual RMS, and a control-distribution z-score.

## Reference target

The first reproducibility target is `Qwen/Qwen2.5-7B-Instruct`.

At upstream Pain-axis commit `4d75cd90e206ea962f7a9101e65c85efea56723b`, the released summary reports:

- 28 decoder layers;
- hidden width 3584;
- best final-token extraction layer 24;
- best mean-pooling extraction layer 26.

Those values are recorded in `configs/models/qwen2.5-7b-instruct.toml`.

## Local setup

Clone this repository and create the Python environment:

```bash
git clone https://github.com/cmoncure/llm-pain-probe.git
cd llm-pain-probe

uv sync --extra hf --extra dev
source .venv/bin/activate
pytest
```

Clone the Pain-axis reference repository at the commit used by this prototype:

```bash
mkdir -p upstream
git clone https://github.com/valen-research/Pain-axis.git upstream/Pain-axis
git -C upstream/Pain-axis checkout 4d75cd90e206ea962f7a9101e65c85efea56723b
```

The released Qwen 2.5 7B Instruct vector is in the reorganized results tree at:

```text
upstream/Pain-axis/results/3.2_pain_vectors/pain_vectors/
  Qwen_2.5_7B_instruct/pain_vectors.pt
```

Convert the upstream PyTorch artifact into the backend-independent safetensors format:

```bash
mkdir -p vectors

python scripts/import_upstream_vector.py \
  upstream/Pain-axis/results/3.2_pain_vectors/pain_vectors/Qwen_2.5_7B_instruct/pain_vectors.pt \
  vectors/qwen2.5-7b-instruct-s2.safetensors \
  --model Qwen/Qwen2.5-7B-Instruct
```

Then run a first raw Hugging Face probe:

```bash
python scripts/smoke_hf_probe.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --vector vectors/qwen2.5-7b-instruct-s2.safetensors \
  --prompt "I made a serious mistake and everyone is blaming me for it."
```

The smoke probe prints one JSON record per recent prompt token containing the raw projection, cosine alignment, and residual RMS. Control-normalized z-scores will be added once the calibration-set runner is implemented.

## Current implementation

- `src/llm_pain_probe/schema.py`: common event schema.
- `src/llm_pain_probe/metrics.py`: projection/cosine/RMS/z-score math.
- `src/llm_pain_probe/vectors.py`: safetensors vector artifact.
- `src/llm_pain_probe/backends/hf.py`: reference post-block residual capture.
- `scripts/import_upstream_vector.py`: Pain-axis artifact importer.
- `scripts/smoke_hf_probe.py`: first end-to-end HF measurement.
- `tests/`: backend-independent and HF hook semantics tests.

## Next milestone

1. Run the Qwen 2.5 7B Instruct smoke probe locally.
2. Verify the observed tensor width/layer semantics against the released model summary.
3. Add control-set calibration and reproduce a published held-out projection result.
4. Add the vLLM backend.
5. Require HF↔vLLM scalar parity before moving to token × layer experiments.
6. Only then add the optimized llama.cpp/GGML scalar probe.

The Pain-axis repository remains an upstream reference implementation rather than vendored project code.
