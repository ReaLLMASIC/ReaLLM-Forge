#!/bin/bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

python3 "$script_dir"/prepare.py -t "zh_cn.txt" --method tiktoken