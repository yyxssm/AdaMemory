#!/usr/bin/env bash


# multi mechaine multi gpu
set -x
CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
EXP_NAME=$3

PY_ARGS=${@:4}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
/opt/conda/envs/PoinTr/bin/python -m torch.distributed.launch \
                                --nnodes=$NNODES \
                                --node_rank=$NODE_RANK \
                                --master_addr=$MASTER_ADDR \
                                --nproc_per_node=$GPUS \
                                --master_port=$PORT \
                                main_memory.py \
                                --config $CONFIG \
                                --exp_name $EXP_NAME \
                                --launcher pytorch --sync_bn ${PY_ARGS}