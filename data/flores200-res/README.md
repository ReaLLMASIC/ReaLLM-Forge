# FLORES-200 restructured data

The existing `get_dataset.sh` downloads text columns from
[muhammadravi251001/restructured-flores200](https://huggingface.co/datasets/muhammadravi251001/restructured-flores200).
The underlying FLORES-200 benchmark is described in the
[FLORES repository](https://github.com/facebookresearch/flores).
FLORES-200 is distributed under [CC BY-SA 4.0](https://github.com/facebookresearch/flores/blob/main/LICENSE_CC-BY-SA); check the pinned repackaged source's
dataset card for any additional provenance/terms before redistribution.

`phoneticize.sh` is an older general-purpose conversion demo. For evaluation of
the TED-trained CJK Char/CharBPE checkpoints, use **`prepare_cjk_eval.py`**. It pins
the source revision, preserves sentence IDs/splits, uses the TED IPA settings,
reuses frozen token IDs, and reports Char OOV coverage rather than replacing
unknown characters. It also accepts offline aligned JSONL.

See [the evaluation guide](../../documentation/cjk_full_checkpoint_eval.md) for
commands, metric definitions, OOV policy and output provenance. Generated
corpora/reports should live outside the Git checkout.
