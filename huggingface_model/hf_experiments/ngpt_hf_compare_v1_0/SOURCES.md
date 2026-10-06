# Primary sources and implementation decisions

Research checked 2026-09-29. The original dense nGPT paper is the normative target.
Links in this file identify the sources; paper/third-party model weights and data
are not bundled.

1. Ilya Loshchilov, Cheng-Ping Hsieh, Simeng Sun, Boris Ginsburg. **nGPT: Normalized
   Transformer with Representation Learning on the Hypersphere.** ICLR 2025.
   Final paper: https://arxiv.org/html/2410.01131v2
   PDF: https://arxiv.org/pdf/2410.01131v2
   Relevant locations: Eqs. 3, 10–11, 15–16, 20–21; Sections 2.5–2.6;
   Appendix A.6, Tables 2–3; Appendix A.7 learning-rate selection.
   Paper page 5 (zero-based PDF page 4) explicitly uses alpha_scale=1/sqrt(d).

2. NVIDIA **nGPT reference implementation**:
   https://github.com/NVIDIA/ngpt
   https://github.com/NVIDIA/ngpt/blob/main/model.py
   https://github.com/NVIDIA/ngpt/blob/main/train.py
   https://github.com/NVIDIA/ngpt/blob/main/launcher.sh
   model.py provides scale initialization and untied embeddings; train.py provides
   the row/column normalization axes and Adam settings; launcher.sh distinguishes
   weight decay and warmup. README warns that its BF16 parameter storage may weaken
   the baseline and overstate speedups, and that the implementation is illustrative.
   This package ports the mathematical recipe, not a byte-for-byte copy of that code.
   The public-repository links are mutable; inspect these files when comparing future
   changes. No unverified commit hash is claimed.

3. Ilya Loshchilov, Boris Ginsburg. **Training nGPT.**
   https://arxiv.org/html/2608.01284v2
   September 1, 2026 revision. Later hybrid-Mamba/MoE training recipe, not implemented
   here. Its GatedAdamW/logit-preconditioning/logarithmic-LR changes must not be
   confused with the original dense-model experiment's Adam/cosine recipe.

4. Hugging Face custom-model API documentation:
   https://huggingface.co/docs/transformers/v4.57.1/en/custom_models
   Configuration/model subclassing, Auto* registration, save_pretrained/from_pretrained.
   Package selects Transformers 4.57.6 (not the current major-version default):
   https://pypi.org/project/transformers/4.57.6/

5. Hugging Face dataset streaming:
   https://huggingface.co/docs/datasets/stream
   Selected Datasets package: https://pypi.org/project/datasets/4.5.0/
   Data source: https://huggingface.co/datasets/Skylion007/openwebtext
   Optional alternate: https://huggingface.co/datasets/JeanKaddour/minipile
   Dataset/tokenizer revisions are resolved to actual immutable Hub SHA values at
   preparation time and recorded along with local token SHA-256 fingerprints.

6. PyTorch wheel selection and SDPA:
   https://pytorch.org/get-started/previous-versions/#v2100
   https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
   Native causal SDPA is used with an explicit scale; there is no custom attention kernel.

No technical implementation decision was taken from an unverified video summary.
The paper equations and NVIDIA's own source code take precedence over secondary
explanations. No claim is made to have watched or transcribed a video.
