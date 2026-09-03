
from pathlib import Path
from collections import Counter
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file="trained_spm_model.model")

def token_freq_for_dir(
    root: str,
    suffixes=(".txt",),
) -> Counter:
    root = Path(root)
    total = Counter()

    for p in root.rglob("*"):
        if p.is_file() and p.suffix in suffixes:
            text = p.read_text(encoding="utf-8", errors="replace")
            ids = sp.encode(text, out_type=int)
            total.update(ids)

    return total

