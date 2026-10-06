# Method and implementation contract — v0.2.0

## Backbone integration

The wrapper reuses the pretrained embedding, norms, projections, MLPs and output
head. Supported model types are `hunyuan_v1_dense`, `llama` and parallel-residual
`gpt_neox`. No backbone parameter is frozen, and no PEFT module is inserted.

Hunyuan applies RoPE first, then learned per-coordinate query/key RMSNorm gains.
Changing that order is incorrect. Its 0.5B head dimension is 128 with 16 query
heads and 8 KV heads, despite hidden_size=1024. The wrapper reads actual attention
head dimensions and uses the native rotary module, including NTK-alpha scaling.
MiniCPM5 likewise uses head_dim=128 despite hidden_size/num_heads being 96.

Before the first eviction, outputs must match the native HF model within floating
point tolerance. This is tested with nonuniform Hunyuan Q/K gains and with GQA.
The unmodified inference baseline uses HF generation itself, not this wrapper.

## State and causal event order

Each layer carries raw pre-RoPE keys, values, and spherical memory directions,
log-radii and validity flags. K/V state has KV-head count. Memory reads repeat the
corresponding KV-head slots over their query groups inside the attention operation.

1. Split input at canonical chunk boundaries determined by total consumed tokens.
2. If the local queue is full, write the oldest whole chunk into memory, then evict
   that chunk. Only already consumed tokens enter this write.
3. Project current hidden states to Q/K/V; append raw keys and values to the queue.
4. Apply native RoPE at positions within the current local queue. For Hunyuan,
   apply its query/key norms afterward. Rebase cached raw keys consistently.
5. Attend causally to local K/V. A rectangular mask permits only old and preceding
   current-chunk positions. Add the gated memory read in head space.
6. Apply the original output projection, residuals and MLP in native order.

Canonical eviction boundaries make results invariant to caller prefill splits.
States are functional; checkpoint recomputation cannot mutate a shared cache.
The cache represents streaming inference. For training, autograd stays connected
through both local K/V and recurrent memory by default. `detach_local=true` is an
explicit truncated-gradient ablation, not the default and not equivalent training.
Finite inference state does not imply constant-memory full-backpropagation training.

After eviction, local positions are rebased. With Hunyuan's learned post-RoPE
per-coordinate gains, this is a real model modification; translation invariance
cannot simply be assumed. Matched local/NLERP/SLERP arms share the same rebasing.

## Spherical memory

Let C be an evicted chunk's token count, D a head's dimension, and M the number of
slots. For each KV head, the writer maps concatenated raw K/V vectors to D-vectors.
Learned unit anchors attend over the C writer outputs to form one candidate per
slot. Its normalized direction is c and its positive magnitude is represented by
log-radius r_c. A slot stores unit direction m and log-radius r_m.

A shared gate network observes (m, c, r_m, r_c, m dot c), with a learned per-slot
bias. Its sigmoid output a lies in [0,1]. Both learned spherical methods receive
the same angular information and have identical parameterization.

- SLERP: update direction along the great-circle arc by fraction a.
- NLERP: normalize (1-a)m + a c.
- Both update log-radius as (1-a)r_m + a r_c.
- EMA: interpolate the unnormalized vectors and recover direction and log-radius.
- Fixed: use a fixed update fraction with the SLERP direction update.

The first nonzero candidate initializes an empty slot. Zero candidates leave it
unchanged. Geometry has stable near-parallel and antipodal handling, exercised by
float64 gradient checks. Reads use learned query/key/value projections and a
separate sigmoid gate initialized to 0.01. Empty memory reads are exactly zero.

Learned-gate NLERP is essential: SLERP and NLERP traverse the same arc with a
reparameterized interpolation fraction. An advantage over a fixed gate would not
alone isolate a benefit of spherical interpolation.

## Optimization and KL

Full FP32 master parameters are updated by AdamW, with BF16 autocast for CUDA
matrix operations. The optional 8-bit optimizer compresses moments only. Memory
geometry is computed in FP32. Activations are checkpointed by layer and vocabulary
loss is computed in chunks of supervised positions; ignored prompt positions do
not allocate vocabulary logits. Next-token targets across chunk boundaries remain
included. Backbone and memory have separate learning rates.

Optional retention uses exact KL(P_reference || P_student), averaged over selected
short-prefix token positions. It is supervised regularization. The separate legacy
GRPO entry point still exists and has a tiny CPU test, but it is not in the main
workflow and has no pretrained CUDA memory/quality validation in this release.

## Reproducibility

The download command locks each model's exact HF commit separately. Every run
records resolved revision, config, package versions, parameter inventory, native
context, actual training lengths, processed tokens, supervised tokens and memory
peaks. Full checkpoints include backbone config/tokenizer, all weights, optimizer,
RNG and progress. Tied embeddings are handled by safetensors' model save/load API.

A native long-context model evaluated under a smaller working window tests memory
compression. It is not evidence of extending beyond its native context.

## Primary implementation references

- https://huggingface.co/tencent/Hunyuan-0.5B-Instruct
- https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/hunyuan_v1_dense/modeling_hunyuan_v1_dense.py
- https://huggingface.co/openbmb/MiniCPM5-1B
- https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/llama/modeling_llama.py
- https://huggingface.co/docs/transformers/v4.51.3/en/model_memory_anatomy
- https://huggingface.co/docs/bitsandbytes/optimizers
