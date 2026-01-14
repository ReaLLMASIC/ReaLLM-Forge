#!/bin/bash
# Note: may need to run this script as `source start_tensorboard.sh` to utilize environment

# Default Port
PORT=6006
ULIMIT=8192
LOG_DIRECTORY=logs

# Check if user provided a specific port
if [ ! -z "$1" ]; then
  PORT="$1"
fi

# Check if user provided a specific port
if [ ! -z "$2" ]; then
  ULIMIT="$2"
fi

# Check if user provided a specific log directory
if [ ! -z "$3" ]; then
  LOG_DIRECTORY="$3"
fi

ulimit -n "$ULIMIT"

echo "ULIMIT=${ULIMIT}; PORT=${PORT}"

tensorboard --logdir="$LOG_DIRECTORY" --port="$PORT" --bind_all --load_fast=false || python3 -m tensorboard.main --logdir="$LOG_DIRECTORY" --port="$PORT" --bind_all --load_fast=false
