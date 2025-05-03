# HellaSwag Evaluation with `sample.py`

1. **Train your model** using `train.py`. Make sure it outputs a checkpoint (e.g., `ckpt.pt`) in the `out` directory (or any directory you choose).

2. **Run HellaSwag evaluation** by using the `--hellaswag_benchmark` flag in `sample.py`. For example:

```bash
python sample.py \
    --hellaswag_benchmark \
    --eval_dataset <your_dataset_name> \
    --few_shot_examples 10 \
    --eval_iters 20 \
    --out_dir out
```

--few_shot_examples represents the number of few-shot examples to prepend to the input prompt for evaluation. These examples help the model learn the format of hellaswag benchmarking and helps it reason more effectively with new examples.

--eval_iters represents the number of evaluation examples we will pick from the hellaswag dataset to test the model with.

--temperature represents the variability in the log probabilities used to score each prompt ending.

--seed for the pseudorandom number generator (mantaining consistent results).