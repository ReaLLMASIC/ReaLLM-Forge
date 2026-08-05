# Modular arithmetic grokking dataset

This dataset recreates the modular addition setup used in classic grokking
experiments: train on a random subset of all ordered pairs `(a, b)` for
`a + b mod p`, hold out the remaining pairs for validation, and repeat the small
training set many times so `train.py` sees a long token stream.

```bash
cd data/modular_arithmetic
python3 prepare.py --modulus 113 --train-fraction 0.3 --train-repeats 200 --val-repeats 20
```

The script writes `train.bin`, `val.bin`, and `meta.pkl` in the same format used
by `data/template/prepare.py` and consumed by `train.py`. It also writes
`train.txt`, `val.txt`, and `manifest.json` for inspection.
