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

## Status

Bootstrap in progress. The Pain-axis repository is treated as an upstream reference implementation rather than vendored project code.
