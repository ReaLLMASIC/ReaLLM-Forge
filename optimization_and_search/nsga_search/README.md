# NSGA-II Search

Minimal utilities for multi-objective hyperparameter/architecture search using NSGA-II within Evo_GPT.

## Files

- **nsga2.py**: core NSGA-II algorithm and operators
- **search_space.py**: search space and variation (mutation/crossover)
- **remote_trainer.py**: training functions via ssh remote connection
- **test.py**: small example/smoke test

## Run Example

* In test.py, set the TODO section [hosts, user, key_filename]
* Ensure the master machine has its public key stored in the slave machines

* The host ips should be defined in host_configs/internal_hosts.yaml
* for an example of the yaml format, see host_configs/hosts.yaml

* From this folder run `bash run_from_scratch.bash`

## Notes

* The conda env on the slave machines are currently hardcoded as "reallmforge" (run dev_env_setup_scripts/00-setup-conda.sh)

