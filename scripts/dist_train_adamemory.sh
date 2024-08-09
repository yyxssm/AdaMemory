#!/usr/bin/env bash

# 原始
set -x
NGPUS=$1
PORT=$2
PY_ARGS=${@:3}


# 原始
python -m torch.distributed.launch --master_port=${PORT} --nproc_per_node=${NGPUS} main_memory.py --launcher pytorch --sync_bn ${PY_ARGS}