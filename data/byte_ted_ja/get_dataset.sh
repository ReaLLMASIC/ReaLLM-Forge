#!/bin/bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

python3 "$script_dir"/prepare.py -t "ja.txt" --method byte