"""Offline synthetic fixture; exercise the exact pipeline on a tiny CPU model."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from .common import ROOT


def create_fixture(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    vocab = {"[UNK]": 0, "[EOS]": 1}
    vocab.update({f"w{i}": i + 2 for i in range(126)})
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    hf = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", eos_token="[EOS]", pad_token="[EOS]", bos_token="[EOS]")
    hf.save_pretrained(output / "tokenizer_source")
    rng = np.random.default_rng(123)
    with open(output / "documents.jsonl", "w") as f:
        for _ in range(500):
            f.write(json.dumps(dict(text=" ".join(f"w{i}" for i in rng.integers(0, 126, 50)))) + "\n")
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="runs/smoke_muon")
    p.add_argument("--preset", default=str(ROOT / "configs/smoke_sweep.json"))
    p.add_argument("--diagnostics", action="store_true")
    a = p.parse_args()
    out = Path(a.output).resolve()
    if out.exists() and any(out.iterdir()): p.error("Smoke output must be a fresh directory")
    create_fixture(out)
    subprocess.run([sys.executable, "-m", "context_sweep.prepare", "--local-jsonl", str(out / "documents.jsonl"),
         "--tokenizer", str(out / "tokenizer_source"), "--output", str(out / "data"),
         "--train-tokens", "10000", "--validation-tokens", "3000", "--validation-permyriad", "2500"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-m", "context_sweep.run", "--preset", a.preset,
         "--data", str(out / "data"), "--output", str(out / "comparison"), "--device", "cpu",
         "--no-monitor"], cwd=ROOT, check=True)
    if a.diagnostics:
        subprocess.run([sys.executable, '-m', 'diagnostics.sweep', '--root', str(out / 'comparison'),
            '--data', str(out / 'data'), '--device', 'cpu', '--tokens', '32'], cwd=ROOT, check=True)


if __name__ == "__main__": main()
