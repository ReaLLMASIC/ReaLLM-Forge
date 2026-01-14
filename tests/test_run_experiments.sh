#!/bin/bash
# test_run_experiments.sh

pushd ../
python3 optimization_and_search/run_experiments.py -c tests/run_optimization_tests/test_range.yaml         --config_format yaml
python3 optimization_and_search/run_experiments.py -c tests/run_optimization_tests/test_lists.yaml         --config_format yaml
python3 optimization_and_search/run_experiments.py -c tests/run_optimization_tests/test_booleans.yaml      --config_format yaml
python3 optimization_and_search/run_experiments.py -c tests/run_optimization_tests/test_param_groups.yaml  --config_format yaml
python3 optimization_and_search/run_experiments.py -c tests/run_optimization_tests/test_conditionals.yaml  --config_format yaml
python3 optimization_and_search/run_experiments.py -c tests/run_optimization_tests/test_nested_param_groups.yaml --config_format yaml
popd
