> Historical v0.1 proposal. Its LoRA settings and planned features are superseded by the v0.2 README and METHOD.md.

**Learned SLERP memory for extending a pretrained model’s usable context**

Experimental and implementation plan · 26 September 2026

**Recommendation.** Retrofit a Hugging Face causal language model with a small recurrent memory. Preserve individual key/value entries for recent tokens; when old entries leave that window, summarize their features and merge them into persistent memory slots using a learned spherical interpolation gate. Every subsequent token can attend to the slots. Apply exactly the same state transition while consuming a prompt and while consuming generated tokens.

Start with `EleutherAI/pythia-410m`, a pretrained GPT-NeoX model whose configuration specifies a 2,048-token context, 24 layers, 16 heads, and hidden width 1,024. This makes 8K/16K/32K tests genuine extensions beyond its original context. Treat this as a mechanism experiment on a small base model, with task-format adaptation before interpreting question-answering scores. [Model configuration](https://huggingface.co/EleutherAI/pythia-410m/blob/main/config.json)

The hardware target is the previously discussed single RTX 4090 with 24 GB VRAM. The settings below are proposed starting points; GPU memory use, speed, and accuracy have not been measured. This deliverable is a plan, not an implemented or trained model.

**1. Define the research claim before training.**

There are three separate lengths to record:

| Quantity | Meaning | First experiment |
|---|---|---|
| Original context, L0 | Context used by the original pretrained checkpoint | 2,048 tokens |
| Adaptation length, Ltrain | Longest prompt plus teacher-forced continuation seen during this experiment’s training | 8,192 tokens |
| Evaluation length, Leval | Total processed sequence length, including generated continuation | 1,024; 2,048; 4,096; 8,192; 16,384; 32,768 |

Success at 8K establishes extension beyond L0. Success at 16K or 32K additionally tests extrapolation beyond Ltrain. Keep these claims separate. A model accepting a 32K input, or generating indefinitely without an out-of-memory error, does not by itself demonstrate useful 32K recall.

The hypothesis is that learned spherical memory updates improve the quality–memory–compute tradeoff of a recurrent adaptation. They cannot preserve every detail of an arbitrarily long sequence in a fixed finite-precision state. The effective horizon will depend on distractors, how many independent facts must survive, and the task.

Related work establishes the broader design space. [Transformer-XL](https://arxiv.org/abs/1901.02860) uses recurrence across segments; [Compressive Transformer](https://arxiv.org/abs/1911.05507) compresses older activations; [Infini-attention](https://arxiv.org/abs/2404.07143) combines local attention with compressive memory. [MiniCache](https://arxiv.org/html/2405.14366v1) already uses spherical interpolation for compression across layers. Our proposed experiment learns updates across **time**, so we should not claim that using SLERP with transformer state is itself new.

**2. Use a recent window and a separate spherical memory.**

Initial configuration:

| Setting | Value |
|---|---|
| Backbone | `EleutherAI/pythia-410m` |
| Recent-token capacity W | 2,048 entries per layer/head, including the current chunk |
| Eviction block C | 256 tokens |
| Memory slots M | 64 per layer/head |
| Memory dimension d | 64, matching this backbone’s head dimension |
| Memory placement | Every attention layer for the first controlled comparison |
| Backbone arithmetic | BF16 |
| Memory states, normalization, angles and gates | FP32 geometry; cast readout as needed |
| Trainable backbone adaptation | LoRA rank 16 on attention input/output projections |
| Initial optimization | AdamW; base weights frozen |
| Initial inference | Batch 1, greedy or ordinary sampling |

The memory state is per input sequence. Learned projections, routing anchors, and gate networks are shared parameters; the memory contents are temporary activations. Reset contents at document/session boundaries. Do not persist a previous evaluation example’s contents into the next one.

For a single layer and head, store M unit directions m_j and positive magnitudes r_j. The original model retains its own normalization, residual structure, biases, and weights. Spherical normalization is introduced inside the memory branch; replacing the pretrained backbone’s normalization would confound the experiment.

**3. Specify the learned update.**

Let i index the C tokens about to be evicted, and j index a memory slot. Let k_i^raw denote a token’s key before rotary position encoding, and v_i its value. Let Ww be a learned projection from their concatenation to a d-dimensional memory feature. Define:

\[
z_i = W_w[k_i^{\mathrm{raw}};v_i].
\]

Let N(x)=x/||x|| for nonzero x. The implementation uses a numerical threshold and handles zero vectors separately. Let s_j be a learned slot-routing anchor and rho a positive learned routing scale. Pool the evicted block separately for each slot:

\[
p_{ji}=\operatorname{softmax}_{i}\left(\rho\,N(s_j)^\top N(z_i)\right),
\qquad c_j=\sum_i p_{ji}z_i.
\]

Only already-processed tokens enter this pool. No future question, future token, or target label enters the writer. The source features are contextual causal states, rather than context-free token embeddings. A later ablation can add explicit within-block position features identically to all memory variants.

Let f be a small gate network shared across slots within a layer/head, b_j a learned slot-specific bias, and sigma the sigmoid function. The update fraction is:

\[
\alpha_j=\sigma\!\left(b_j+f\left[m_j;N(c_j);\log r_j;\log\|c_j\|;m_j^\top N(c_j)\right]\right).
\]

Thus the model learns both which old-token features belong in a slot and how strongly to replace its existing contents. Initialize different slot groups with different gate biases, for example fractions near 0.5, 0.1, 0.03, and 0.01. These are starting timescales, not guarantees about retention.

For two unit vectors u and v, define theta=arccos(u^T v). Away from the degenerate parallel/antipodal cases:

\[
\operatorname{SLERP}(u,v,\alpha)=
\frac{\sin((1-\alpha)\theta)}{\sin\theta}u+
\frac{\sin(\alpha\theta)}{\sin\theta}v.
\]

Update a slot with:

\[
m_j^{\mathrm{new}}=\operatorname{SLERP}(m_j,N(c_j),\alpha_j),
\]

\[
\log r_j^{\mathrm{new}}=(1-\alpha_j)\log r_j+\alpha_j\log\|c_j\|.
\]

An empty slot is initialized directly from its first valid candidate. An approximately zero candidate skips the write. Magnitude is tracked separately because normalizing a vector removes its original norm. Also run a fixed-radius ablation to determine whether this scalar is useful.

For reading, let q_t^raw be the current token’s query before RoPE. Introduce learned memory projections Wq, Wk, and Wv, and a positive learned attention scale tau. Define:

\[
q_t^M=N(W_q q_t^{\mathrm{raw}}),\quad
k_j^M=N(W_k m_j),\quad
v_j^M=W_v(r_jm_j).
\]

\[
a_{tj}=\operatorname{softmax}_j(\tau(q_t^M)^\top k_j^M),\qquad
o_t^M=\sum_j a_{tj}v_j^M.
\]

Let o_t^local be the original per-head local-attention output. A learned scalar read gate g_t in [0,1] blends in the memory contribution:

\[
o_t=o_t^{\mathrm{local}}+g_t o_t^M.
\]

Concatenate heads and apply the backbone’s existing attention output projection. Initialize g_t near 0.01; an empty memory returns exactly zero. Normalize only the memory query/key branch and initialize tau near sqrt(d). These are new branch parameters, not changes to the native attention scaling.

This choice uses SLERP for the **persistent update of representations of prior tokens**. It does not interpolate vocabulary IDs or assume that an interpolated vector represents one exact recoverable token.

**4. Keep the state transition causal and identical during reading and generation.**

Maintain an absolute consumed-token counter, a bounded local queue, and the memory bank. A 256-token block boundary is determined by the sequence, never by the size of a Python call or the end of a prompt.

Before processing a new token when the local queue is full:

1. Select its oldest C entries in every layer.
2. Update that layer’s memory using only those entries.
3. Remove those entries from the local queue.
4. Process up to C new tokens with causal attention and the updated memory.

For example, after consuming tokens 0 through 2,047, compress entries 0 through 255 before consuming token 2,048. The local queue then contains entries 256 through 2,047. During the next block its occupancy increases from 1,792 to 2,048. This blockwise schedule is slightly different from a strict 2,048-token sliding window; every matched local/memory baseline must use the same schedule.

When decoding, a generated token is consumed on the next forward pass, just like a prompt token. A prompt ending halfway through a block does not flush or reset the memory. Arbitrarily sized prefill calls must split internally at the same logical boundaries.

Do not compress an entire current chunk first and expose that summary to earlier tokens of the same chunk: that would leak the future during teacher forcing. Padding positions never enter pooling or advance a stream’s token counter.

**RoPE handling.** Store local keys before rotation, then apply native RoPE at rebased positions in the current local queue. Rebase both retained keys and new queries consistently after eviction. The same offset applied to both preserves relative rotary distances. Pythia uses partial rotary dimensions; preserve that subset exactly. Mixing keys already rotated at incompatible positions would contaminate the memory with phase. The memory branch therefore operates on unrotated features and has its own content-based readout. [GPT-NeoX implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py)

For an optimized inference implementation, one can instead rotate retained cached keys by the common offset when eviction happens. First implement the clearer raw-key reference version and verify that optimization against it.

**5. Treat normalized interpolation as an essential control.**

For non-antipodal unit endpoints, SLERP and normalized linear interpolation follow the same minor great-circle arc. Their interpolation parameters differ. Define:

\[
\beta=\frac{\sin(\alpha\theta)}{\sin((1-\alpha)\theta)+\sin(\alpha\theta)}.
\]

Then:

\[
\operatorname{SLERP}(u,v,\alpha)=N((1-\beta)u+\beta v).
\]

This follows by taking the ratio of the two positive SLERP coefficients; their common scale disappears under normalization. A learned angle-aware gate can therefore reproduce the same path with normalized interpolation. A possible SLERP gain is about parameterization, optimization, or inductive bias—not automatically greater representational capacity.

Compare these update rules with identical pooling, readout, magnitude handling, slot count, and training budgets:

| Update | What it tests |
|---|---|
| Learned ordinary vector EMA | Whether the spherical constraint matters |
| Learned normalized interpolation | The main spherical baseline; include angle in its gate input |
| Fixed-fraction SLERP | Whether learning the update gate matters |
| Learned SLERP | The proposed method |

The ordinary EMA arm should update the full magnitude-bearing vector and then expose the same direction/magnitude readout, so that a different reader is not the explanation for a result. Also keep a SLERP-equivalent, analytically remapped normalized-interpolation implementation as a numerical check rather than counting it as an independent method.

Numerical rules: calculate geometry in FP32; use normalized interpolation for nearly parallel endpoints; use a tangent-form evaluation near antipodal endpoints. At exact antipodes the arc is nonunique, so use a deterministic orthogonal tangent and log how often that branch is used. Do not flip the sign of an embedding merely to shorten the arc; antipodal embeddings are not interchangeable quaternions. Verify endpoints, norm preservation, and finite gradients, including near-degenerate inputs.

**6. Build on Hugging Face with explicit custom model/state code.**

| Component | API and responsibility |
|---|---|
| Original weights/tokenizer | `AutoModelForCausalLM.from_pretrained`, `AutoTokenizer.from_pretrained` |
| Backbone adapters | PEFT `LoraConfig` and `get_peft_model` |
| Documents | Datasets `load_dataset(..., streaming=True)` |
| Training | Accelerate loop with explicit recurrent state and loss accounting |
| Serialization | Custom model/config compatible with `save_pretrained` and `from_pretrained` |
| Inference | Custom bounded cache and model generation integration |
| Optional RL | TRL `GRPOTrainer` extension with state-aware rollout and scoring |

The public APIs provide the surrounding machinery. `SlerpMemory`, `SlerpState`, and the modified GPT-NeoX attention are proposed project components that must be implemented; Hugging Face does not expose this proposed architecture as a configuration flag. Its [custom-model APIs](https://huggingface.co/docs/transformers/en/custom_models) support the serialization pattern.

An illustrative loading/adaptation fragment, to run only after the custom module is implemented:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from slerp_context import attach_slerp_memory  # proposed project code

model_id = "EleutherAI/pythia-410m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
base = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.bfloat16, attn_implementation="sdpa"
)
model = attach_slerp_memory(
    base, window_size=2048, eviction_size=256,
    slots_per_head=64, memory_dim=64,
)
model = get_peft_model(model, LoraConfig(
    task_type="CAUSAL_LM", r=16, lora_alpha=32,
    target_modules=["query_key_value", "dense"],
    modules_to_save=["slerp_memory"],
    lora_dropout=0.0, bias="none",
))
```

Name each inserted trainable memory module `slerp_memory`, and inspect the actual matched module paths and trainable-parameter inventory. `modules_to_save` is needed so the additional trained modules accompany the adapters. Check a save/reload round trip before expensive training. [PEFT LoRA configuration](https://huggingface.co/docs/peft/en/developer_guides/lora)

Preserve the original pretrained context metadata and record `local_window`, `train_sequence_limit`, and evaluation lengths separately. A larger integer in `max_position_embeddings` does not implement memory or demonstrate recall. Resolve and record exact model/dataset commit SHAs and package versions during setup. Current HF RoPE documentation uses `rope_parameters`; the separate position-interpolation baseline must use the schema supported by the pinned version. [RoPE utilities](https://huggingface.co/docs/transformers/en/internal/rope_utils)

Use a functional state object for training: forward takes state_in and returns state_out, without in-place writes to tensors needed for backward. Use a cache adapter only for inference. Standard Transformers caching is documented as an inference mechanism; it is not a substitute for a differentiable recurrent training state. Track total consumed length separately from physical cache occupancy, and override input slicing/mask preparation accordingly. [Cache documentation](https://huggingface.co/docs/transformers/en/cache_explanation)

A custom attention backend alone is insufficient because the writer needs pre-RoPE features and owns recurrence. Modify the model’s attention module at that point. If registering an attention backend, register its mask handling too; otherwise the API can omit mask construction. Rectangular cached attention needs explicit causal alignment between current queries and prior/current keys. [Attention backend documentation](https://huggingface.co/docs/transformers/en/attention_interface)

During teacher forcing, compute next-token loss across chunk boundaries, including the last logit of one chunk against the first token of the next. Do not silently drop these targets. During generation, process each input token exactly once. For the initial release, explicitly reject unsupported beam/speculative-cache operations rather than approximating their rollback/reordering semantics.

Suggested implementation units and acceptance criteria:

| Proposed unit | Required behavior |
|---|---|
| `geometry.py` | Stable SLERP, NLERP, magnitude updates; degenerate-case gradients |
| `memory.py` | Slot pooling, read/write gates, masked empty-state handling |
| `modeling.py` | Attention integration before RoPE; unchanged native residual structure |
| `state.py` | Functional training state; inference cache; separate local/logical positions |
| `data.py` | Whole-document streams, synthetic episodes, leak-free held-out generators |
| `train.py` | Continued pretraining/SFT, recurrent gradients, bounded checkpoints |
| `grpo.py` | Optional rollout/replay with correct policy-specific memory |
| `evaluate.py` | Length/distance/capacity/position/generation tests and raw outputs |

**7. Start with supervised training and explicit long-distance credit assignment.**

The gate is differentiable. Next-token and answer losses can train it directly; there is no need to introduce RL merely because it makes a memory decision.

| Stage | Data/length | Trainable parameters | Initial budget per arm |
|---|---|---|---|
| Mechanical validation | Tiny deterministic episodes | None required | Before training |
| Memory warm-up | 1K–2K sequences with W=512, C=256 to force compression | Memory modules only | 2M processed tokens |
| Continued pretraining | Mix lengths through 8K; restore W=2,048 | Memory + LoRA | 20M processed tokens |
| Memory-focused SFT | Delayed retrieval, updates, multiple queries, tracing; through 8K | Memory + LoRA | 10M processed tokens |
| Optional GRPO | Verifiable delayed-answer episodes through 8K | Memory + LoRA at lower LR | 200–500 initial updates |

First run a smaller approximately 5M-token pilot per principal arm. Scale to the table’s approximately 32M-token supervised budget only if there is a measurable memory benefit and a healthy gradient path. Token budgets include prompt processing, not only loss-bearing answer tokens. Report scored tokens separately.

Suggested optimizer starting points: memory LR 3e-4 during warm-up and 1e-4 afterward; LoRA LR 5e-5; AdamW weight decay 0.01 on suitable matrices, zero on scalar gates/biases; gradient clipping at 1.0; 3% warm-up. Use batch size 1 and accumulation to about 32K processed tokens per optimizer update. Treat these as tunable validation choices, not established optimal values.

For the continued-pretraining mix, start with approximately 60% natural-document episodes and 40% synthetic memory episodes. Retain short examples throughout. A useful length mixture has approximately 20% short sequences and the remainder split between 4K and 8K, while reporting the resulting token-weighted distribution.

For natural text, stream a bounded subset of the Parquet [emozilla/pg19](https://huggingface.co/datasets/emozilla/pg19) export with `load_dataset("emozilla/pg19", split="train", streaming=True)`, adding an exact pinned revision in the actual run. Verify its book IDs/splits against the original [PG-19 dataset](https://huggingface.co/datasets/deepmind/pg19); do not assume a mirror is byte-identical. Use book-level train/validation/test separation. The original repository uses a loading script, so the Parquet route avoids relying on legacy dataset-script support. [Dataset streaming API](https://huggingface.co/docs/datasets/en/stream)

Public books may overlap original pretraining data. Therefore PG-19 is a natural-language diagnostic, not our strongest evidence against memorization. Fresh synthetic facts and held-out generated episodes provide that evidence.

Synthetic tasks should include unpredictable key/value bindings; later corrections to the same key; a question revealed only after the document; several questions about different earlier spans; simple variable chains; and selection/aggregation across distant records. Vary distractor text, key/value tokenization, fact count, and distance. Hold out generator seeds, identifiers, and template families. Include examples whose answer-bearing facts have actually left the local queue.

Let L_lm be mean next-token cross-entropy on natural documents, L_answer be mean answer-token cross-entropy on synthetic tasks, and L_short be KL divergence from the frozen original model on short natural examples. A starting mixed objective is:

\[
L = L_{lm}+2L_{answer}+0.1L_{short}.
\]

Average each term over its own valid tokens before applying weights, and define absent terms as zero. The teacher sees only sequences within its native context. Do not use the original 2K model as an oracle for 16K reasoning. On answer-only tasks, mask prompt labels, but preserve differentiability through prompt memory construction.

**Gradient requirement.** Keep the memory recurrence differentiable across the entire 4K/8K episode. Local K/V features can be detached at chunk boundaries to limit backbone history; the memory writer consumes those detached features through its trainable Ww. This is a deliberate truncated gradient path through local cached features, while the chain of memory updates remains connected to delayed answer loss. Do not detach memory every 256 tokens, which would prevent direct delayed supervision of the early write.

Frozen backbone weights still need differentiable forward operations when they transmit gradients from logits to the memory branch. Wrapping the whole backbone in `no_grad()` would break that path. Use checkpointed, pure chunk computations and chunked LM-head loss. Update optimizer weights only after the complete episode’s backward pass. Verify that a late answer produces a nonzero gradient at an early memory write. If full-episode graphs exceed VRAM, implement checkpoint/replay of the recurrence or reduce the pilot length; do not silently change the experiment into short-horizon truncated training.

A small from-scratch companion experiment can later isolate architecture effects: initialize an identical small GPT-NeoX backbone for each memory rule, train only to 2K–4K, then test longer. It is secondary to the requested pretrained-model finetuning path.

**8. Add GRPO only after supervised retrieval works.**

The RL target is a correct delayed answer, not a geometrically attractive memory state. Begin with four sampled answers per prompt, temperature 1, no top-k/top-p truncation for the initial likelihood implementation, and at most 64 new tokens. Use exact-match rewards on passkeys; on structured multi-hop tasks, use 0.8 for a correct answer and 0.2 for a verified supporting record/path. Incorrect or invalid answers must not earn a formatting-only reward.

Use group-normalized rewards, an initial clipping range of 0.2, and a small KL penalty against a frozen supervised checkpoint, for example coefficient 0.02. Start with LR around 1e-5. These are initial experimental settings. Log the fraction of groups with differing rewards: when every sample scores zero or every sample scores one, the group supplies little useful learning signal. Adjust task difficulty using training/validation data rather than adding arbitrary rewards.

TRL provides `GRPOTrainer` and an experimental `rollout_func` hook. Our recurrent model needs matching rollout **and differentiable likelihood replay**; a rollout hook alone is not enough. [GRPO documentation](https://huggingface.co/docs/trl/en/grpo_trainer)

During sampling, clone prompt state per completion so siblings cannot contaminate each other. During policy updates, rebuild memory from the raw prompt under the policy being scored. A detached state created by the old policy is not the current policy’s differentiable prefix. The frozen reference also needs its own memory computation and a snapshot of the trained memory parameters, not merely disabled LoRA weights. Prompt lengths must be asserted before and after collation to prevent silent truncation.

Gradients from completion log probabilities can train the differentiable writer through the replayed prefix. Freezing or detaching that prefix would restrict RL mostly to reading/output behavior. Delay hard slot selection and vLLM integration until the differentiable reference path is validated.

Compare SFT alone, SFT plus additional supervised updates, and SFT plus GRPO under matched additional compute/tokens as closely as possible. A gain after RL is not evidence for the RL objective if the comparison simply received less training.

**9. Evaluate memory use, extension, and generation separately.**

| Evaluation | Required measurement |
|---|---|
| Delayed random key/value recall | Exact match against distance beyond W and total length |
| Multiple needles/facts | Exact match and set F1 as independent fact count increases |
| Latest-value updates | Whether old values are overwritten correctly |
| Multi-hop tracing/aggregation | Distant records combined correctly |
| Evidence-position sweep | Early, middle and late placement at matched difficulty |
| Query-last versus query-first | Whether the writer works before it knows the question |
| Natural-document prediction | NLL/perplexity on identical target suffixes with nested prefix lengths |
| Long generated continuation | Recall of earlier facts after more than W generated tokens |
| Native-context regression | Short-task accuracy and short-text perplexity |
| Systems measurements | Peak allocated/reserved VRAM, prefill/decode throughput, latency, bytes of recurrent state |

Use the official [RULER](https://github.com/NVIDIA/RULER) generator/evaluator for the standardized synthetic benchmark component. RULER includes retrieval, multi-hop tracing and aggregation; its motivation is precisely that one easy needle test can overstate usable context. Run the official task set where feasible and explicitly label any altered templates or subset. Keep task-learning warm-up data distinct from final benchmark templates. [RULER paper](https://arxiv.org/abs/2404.06654)

A 410M base model may fail a task even with a short context. Establish an easy native-length task floor first; otherwise an all-zero long-context result cannot distinguish poor reasoning from poor memory. Report every standard score rather than selecting only tasks that succeed.

For generated-history tests, separately report teacher-forced and free-running continuation. One free-running protocol is to record an identifiable fact in the model’s early output, generate over 2,048 subsequent tokens, and only then ask for the original fact. Score both correctness against the task’s intended value and consistency with the actually emitted value. Exclude or separately tag cases where the fact was repeated into the recent window; otherwise the test can be passed without distant retention. Do not inject the final query early into the memory writer.

Natural-language perplexity often improves without solving retrieval. Score identical final 512-token suffixes after 2K/4K/8K/16K/32K nested prefixes, plus whole-stream NLL; distinguish tokenizer-level NLL from PG-19’s published word-level normalization. Reset state between books and examples.

For capacity, vary the number of independent facts through values such as 1, 8, 32, 128, and 512. Test M in {16,64,256}, initially on validation. Slot count is not a literal count of perfectly storable facts. Plot accuracy against both token distance and fact count.

Use three training seeds for the final main comparison. Save per-example outputs and bootstrap confidence intervals over examples, alongside the variation across training seeds. Use paired examples across methods. Stratify memory-dependent examples where the relevant facts have already been evicted, instead of allowing many recent-window examples to dominate the average.

**10. Compare against fair baselines.**

| Arm | Purpose |
|---|---|
| Original checkpoint within 2K | Native behavior reference |
| Same trained backbone/adapters with blockwise local attention only | Isolate value of the explicit memory architecture |
| Local attention plus initial attention sinks | Stronger streaming stability control |
| RoPE position interpolation plus finetuning | Conventional full-attention context extension |
| Same memory architecture with learned EMA | Test normalization versus ordinary updating |
| Same memory architecture with learned NLERP | Main geometric/optimization control |
| Fixed SLERP gate | Test learned forgetting/retention |
| Learned SLERP | Proposed method |

For the first inexpensive pilot, prioritize local-only, learned NLERP, and learned SLERP; include the position-interpolation arm before making a broader context-extension claim. [Position interpolation](https://arxiv.org/abs/2306.15595) and [YaRN](https://arxiv.org/abs/2309.00071) are established RoPE extension approaches. Full attention has a different memory/compute budget, so report both equal-data results and resource costs rather than calling it an equal-memory comparison. [StreamingLLM](https://arxiv.org/abs/2309.17453) motivates the attention-sink control; streaming stability alone does not establish recovery of discarded arbitrary facts.

For a PI baseline trained through 8K from L0=2K, use a documented factor-4 training configuration with the native partial rotary dimensions preserved. Evaluate its fixed trained configuration at 16K/32K as extrapolation; optionally also report separately labeled inference-only rescaling at those targets. Do not combine results from different scaling factors into one supposedly fixed-model curve.

Compare memory variants with equal trainable capacity, data order, length distribution, token budgets, optimizer settings, and native normalization. All local baselines use the same block boundaries/rebasing. Report additional parameter counts and compute for memory arms. Include both branch-zeroed inference with otherwise fixed local state and a full run with memory disabled: local contextual K/V can carry indirect information from earlier memory reads, so a single late ablation is not a complete intervention.

**11. Predefine decision criteria and verification gates.**

The initial research decision is whether a bounded memory helps at all, then whether SLERP helps beyond normalized interpolation. Suggested thresholds to preregister as experiment goals:

- At 8K and 16K, at least a 10 percentage-point gain over the matched local-only arm on a predefined aggregate of memory-dependent synthetic tasks, with paired uncertainty estimates.
- On simple recall, retain at least 90% of the adapted model’s 2K accuracy, only interpreting that ratio when its 2K accuracy is at least 80%.
- Keep short-text perplexity degradation below 5% relative to the native/adapted short-context reference specified in advance.
- For the SLERP-specific claim, demonstrate a repeatable gain over the learned angle-aware NLERP arm; otherwise report that spherical memory helped but SLERP did not establish a separate advantage.
- Record 16K/32K performance with no training above 8K. Any later 16K finetuning starts a separately labeled experiment.

These are decision thresholds, not predicted results. Show the complete curves and absolute scores, even when a threshold is missed.

Before training, require future-token perturbation not to change earlier logits; chunked prefill and token-by-token inference to agree within numerical tolerance under the same logical schedule; exact save/load continuation of a session state; no cross-document or cross-completion state leakage; proper padding masks; finite geometry gradients; and a verified delayed gradient at an early write. With memory empty/disabled and zero-initialized LoRA, the model should reproduce the original short-context computation within backend numerical tolerance. Check this separately from the ordinary sliding-window approximation after eviction.

**12. Keep the 4090 run and disk use bounded.**

For L layers, H attention heads, head width d, and BF16 local K/V, the persistent local-state size is approximately:

\[
2LHWd\times2\ \mathrm{bytes}.
\]

At L=24, H=16, W=2,048, d=64 this is 192 MiB. One FP32 spherical bank adds LHMd times 4 bytes, or 6 MiB at M=64, plus small magnitude/validity tensors. A separately retained rotated-key buffer or materialized memory K/V adds more. These are **inference-state calculations only**: parameters, gradients, optimizer states, activations, logits, workspaces and training graphs remain additional costs.

Bounded W and M make recurrent state independent of total processed length. Ignoring fixed model projection costs, attention and writing scale approximately linearly with total length T: local attention O(T W), memory reads O(T M), and block pooling amortizes to O(T M), per layer/head and dimension factors. The sequential update schedule can still reduce throughput; benchmark it instead of inferring speed from big-O complexity.

Use BF16 backbone weights, FP32 geometry, SDPA where the required mask permits an efficient kernel, gradient checkpointing, chunked vocabulary loss, and a single sequence per microbatch. Profile the first 100–200 steps and extrapolate runtime from measured tokens/second. If masks force a slower backend, record it rather than assuming FlashAttention speed. Select smaller pilot lengths or replay checkpointing if needed.

Store one base-model snapshot, bounded token shards, adapter-plus-memory checkpoints, optimizer/RNG/data-position state for the latest resumable checkpoint, one best checkpoint, and raw evaluation outputs. Do not write a full merged model at each save. Target under 20 GB of incremental project data with a configurable cap, leaving headroom if only about 30 GB is free; environment/package caches need their own allowance. No full corpus download is required.

Save run configuration, base/dataset/package revisions, processed/scored token counts, trainable-parameter inventory, seed, and exact evaluation generator settings. Raw outputs should contain example ID, split, task, input/total length, evidence positions, evicted/not-evicted status, distance, fact count, output, target, score, state size, latency and peak memory. From those records generate accuracy-versus-length, accuracy-versus-distance, position heatmaps, capacity curves, NLL curves, and throughput/VRAM comparisons.

The recommended implementation sequence is: validate the state machine and geometry; establish a local-only baseline; run the three-arm 5M-token pilot; add the conventional PI comparison; expand successful supervised arms to the full budget; then assess whether GRPO improves on an equal-budget supervised continuation. Replication on `Qwen/Qwen2.5-0.5B` can follow, but its native configuration is already 32,768 tokens, so genuine extension there should test 64K/128K rather than relabeling an artificially shortened window. [Qwen configuration](https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json)

**Verification performed while preparing this plan.** The linked model configurations and API documentation were inspected. A NumPy numerical check over 5,000 random 64-dimensional endpoint pairs confirmed SLERP norm preservation to floating-point precision and its equivalence to analytically remapped normalized interpolation (maximum coordinate error approximately 2.2e-16 in float64). The state-size arithmetic above was computed directly. No pretrained weights were downloaded and no GPU training or end-to-end integration was run.
