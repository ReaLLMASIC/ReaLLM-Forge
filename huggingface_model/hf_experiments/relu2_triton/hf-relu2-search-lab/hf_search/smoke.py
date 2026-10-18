"""Offline CPU fixture for all three independent-head HF model arms."""
import argparse
from pathlib import Path
import subprocess
import sys
from experiment.smoke import create_fixture
from experiment.common import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='runs/hf_search_smoke')
    a = p.parse_args()
    out = Path(a.output).resolve()
    if out.exists() and any(out.iterdir()): p.error('Choose a fresh output directory')
    create_fixture(out)
    subprocess.run([sys.executable,'-m','context_sweep.prepare','--local-jsonl',str(out/'documents.jsonl'),
        '--tokenizer',str(out/'tokenizer_source'),'--output',str(out/'data'),
        '--train-tokens','10000','--validation-tokens','3000','--validation-permyriad','2500'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-m','hf_search.launch','run','--preset',str(ROOT/'configs/hf_search_smoke.json'),
        '--data',str(out/'data'),'--output',str(out/'comparison'),'--device','cpu'],cwd=ROOT,check=True)


if __name__ == '__main__': main()
