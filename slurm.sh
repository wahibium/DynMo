#!/bin/bash
# Job name:
#SBATCH --job-name=bert
#
# Account:
#SBATCH --account=msoyturk
#
# Number of nodes:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --partition=dgx2q

#
# Wall clock limit:
#SBATCH --time=20:00:00
#

NUMBER_OF_LAYERS=${1:-24}
BALANCER=${2:-diffusion}
GPUS_PER_NODE=${3:-8}
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=${4:-1}
NUM_GPUS_AFTER_PACKING=${5:-6}
FINAL_SPARSITY=${6:-0.9}

## Command(s) to run (example):
module load cuda11.3/toolkit/11.3.0
module load ninja-1.10.0
conda activate myenv

echo "NODELIST="${SLURM_NODELIST}
echo "starts"
LD_PRELOAD=/home/msoyturk/workspace/sputnik/build/sputnik/libsputnik.so bash examples/pretrain_bert_distributed_with_mp.sh $NUMBER_OF_LAYERS $BALANCER $GPUS_PER_NODE 1
echo "ends"
