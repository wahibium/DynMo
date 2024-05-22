#!/bin/bash

NUMBER_OF_LAYERS=${1:-24}
BALANCER=${2:-diffusion}
GPUS_PER_NODE=${3:-4}
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=${4:-1}
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=data/my-new-bert_text_sentence
CHECKPOINT_PATH=checkpoints/bert_distributed
VOCAB_FILE=data/bert-large-uncased-vocab.txt

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

#python -m torch.distributed.launch
LD_PRELOAD=/home/msoyturk/workspace/sputnik/build/sputnik/libsputnik.so torchrun $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size $WORLD_SIZE \
       --num-layers $NUMBER_OF_LAYERS \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 8 \
       --global-batch-size 64 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters 1000000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --min-lr 1.0e-5 \
       --lr-decay-iters 9900 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .005 \
       --log-interval 10 \
       --save-interval 100000 \
       --eval-interval 100000 \
       --eval-iters 100000 \
       --sparsify \
       --pruning-type gradual \
       --pruning-start-iter 2 \
       --pruning-end-iter 10 \
       --pruning-freq 2 \
       --final-sparsity 0.9 \
       --load-balance \
       --balancer $BALANCER \
       #--packing \
       #--packed-num-gpus 2 \