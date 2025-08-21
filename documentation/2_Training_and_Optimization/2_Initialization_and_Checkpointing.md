# 2.2 Initialization & Checkpointing

These arguments control how the training run is initialized and how checkpoints are saved.

## Initialization

| Argument | Type | Default | Description |
|---|---|---|---|
| `--init_from` | str | 'scratch' | The initialization method. 'scratch' trains from a random initialization. 'resume' resumes from a checkpoint in `out_dir`. 'prev_run' resumes from a checkpoint in a different directory. 'gpt2' starts from a pretrained GPT-2 model. |
| `--gpt2_type` | str | 'gpt2' | The type of GPT-2 model to use when `init_from` is 'gpt2'. Options are 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'. |
| `--prev_run_ckpt` | str | '' | The directory of the previous run to resume from when `init_from` is 'prev_run'. |
| `--init_from_ckpt` | str | 'ckpt.pt' | The name of the checkpoint file to use when resuming. |

## Checkpointing

| Argument | Type | Default | Description |
|---|---|---|---|
| `--save_major_ckpt_interval` | int | None | If specified, saves a major checkpoint every this many iterations, named with the iteration number. This is useful for keeping a history of the model's progress. |
| `--only_save_checkpoint_at_end` | bool | False | If `True`, only saves the final checkpoint at the end of training. This can save disk space. |
| `--always_save_checkpoint` | bool | False | If `True`, saves a checkpoint at every evaluation interval, regardless of whether the validation loss has improved. This is useful for debugging. |
| `--never_save_checkpoint` | bool | False | If `True`, disables saving of all checkpoints. This is useful for quick experiments where you don't need to save the model. |