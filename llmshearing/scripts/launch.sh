#!/bin/sh
PROJ_DIR=/global/cfs/cdirs/m4645/alvinliu/repo/LLM-Shearing

# num_nodes=$(scontrol show job $SLURM_JOB_ID | grep NodeList=della | wc -l)
num_nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l)
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export MASTER_ADDR=$master_addr
echo $SLURM_GPUS_PER_NODE

export WORLD_SIZE=$(( $num_nodes * $SLURM_GPUS_PER_NODE ))
export MASTER_PORT=$(( 10000 + RANDOM % 10000 ))

echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE
echo "num_nodes="$num_nodes

export HF_HOME="/pscratch/sd/a/alvinliu/datasets/.cache/huggingface"

if [[ $num_nodes == 1 ]]; then shifter composer $PROJ_DIR/llmshearing/train.py $@; 
else srun shifter bash $PROJ_DIR/llmshearing/scripts/srun_launch.sh $@; fi
# else srun -u shifter torchrun --nproc-per-node=4 --master-port=$MASTER_PORT --nnodes=$SLURM_JOB_NUM_NODES \
#     --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT $PROJ_DIR/llmshearing/train.py $@; fi
 
