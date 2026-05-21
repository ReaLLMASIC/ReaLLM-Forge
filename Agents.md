# Agents.md

## Purpose
This file gives coding agents the minimum repository-specific rules needed to complete tasks correctly.

## Scope
- Applies to the entire repository unless a deeper `Agents.md` overrides it.

## Must-Follow Rules
- Required behavior for fixes:
  - Add or update tests for bug fixes/features when applicable.
  - Keep changes minimal, task-focused, and aligned when possible with the
    spirit/structure of the repo.

## Important Core Files
- `model.py`: `model architecture definition, defined here and imported files
  from variations, etc.`
- `gpt_conf.py`: `contains settings for the model.py architecture`
- `train.py`: `central file for any conventional transformer training paradigm`
- `train_args.py`: `central file for args for train.py and other training paradigm files`

## Visualization Files
- `run_exploration_monitor.py`: `creates a tui for exploring yaml files created
  by optimization_and_search/run_experiments.py and may need to be updated when
  run_experiments.py is updated.`
- `view_hp_log.py`: `for monitoring and exploring hyperparam_search.py searches..`

## Important Directories
- `variations/...`: `The proper location for model architecture related
  variations. Typically one should check if additions or modifications here may
  require updating the gpt_conf.py file or the train_args.py as well.`
- `train_variations/...`: `the proper location for training related variations,
  e.g. optimizers, loss_function variations, etc.`
- `explorations/...`: `Location for yaml based grid searches or exploration
  specifications. See default_inf.yaml for a template.  As much as possible
  utilize common_group to minimize length of name of output files. Also for ease
  with grouped settings, utilize the named_static_groups and parameter group
  with named_group_static for compact specifications. named_group-alternate
  allows for varying across named_group_static or named_variation_groups.`
- `demos/...`: `high level bash scripts for full end-to-end demos of new
  features if requested. See other scripts in this folder for reference.`
- `data/...`: `Adding scripts for new datasets will go into this folder. Each of
  the new folders should have softlinks to the data/template/utils and
  data/template/prepare.py. There should be a script called get_dataset.sh in
  any dataset folder, which when run will go through creation of input.txt file
  ready for tokenization with prepare.py.  There should be a brief readme in the
  new dataset folder describing the dataset, a link to the original source, as
  well the license of the original dataset.`
- `data/template/...`: `This will have any tokenizer related changes.
  data/template/nanogpt_tokenizers.py will have tokenizer definitions
  (the file is intentionally NOT named tokenizers.py so it does not shadow
  the third-party HuggingFace tokenizers package on sys.path),
  data/template/prepare.py will be how one calls the tokenizer and selects
  argparse settings, and data/template/tests.py has tests for each of the
  tokenizer definitions. When adding new features to the tokenizer, or new
  tokenizers, ensure to add new tests as well to data/template/tests.py.`
- `huggingface_model/...`: `this and subfolders will contain scripts that
  directly build from models from huggingface.`
- `hp_searches/...`: `This subfolder contains tests and saved search settings for
  the hyperparam_search.py script in the main repo folder. When creating
  modifications to the hyperparam_search.py script or the view_hp_log.py script,
  ensure to create a new test .sh script and yaml file in the hp_searches
  directory.`
- `optimization_and_search/...`: `This folder should contain any optimization
  and search related script, and configuration settings for ip_addresses to
  utilize for the search when in distributed mode. Scripts such as
  run_experiments.py and run_from_yaml.py run grid searches on the explorations/
  yaml file specified. We also have a genetic algorithm search which has the
  distributed search implemented. Note: hyperparam_search.py should belong in
  here, and we still need to add a distributed machine solution to
  hyperparam_search.py and the relevant optimization scripts in this directory.`

## Validation Gate (before finishing)
- [ ] All required commands pass locally.
- [ ] No unrelated refactors.
- [ ] Do not commit or check in any binaries or image files.

## PR/Commit Notes
- Commit title format: `<type(scope): summary>`

