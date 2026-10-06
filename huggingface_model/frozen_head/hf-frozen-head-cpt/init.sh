#!/bin/bash

python -m pip install -r requirements-benchmarks.txt
bash run_task_h100.sh configs/h100_math_pilot.json
