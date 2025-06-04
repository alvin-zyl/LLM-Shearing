#!/bin/sh
#SBATCH -A m4788_g
#SBATCH -q premium
#SBATCH -C gpu&hbm80g
#SBATCH -t 1:00:00
#SBATCH --image=alvinliu12138/zhanggroup:shearllm
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1

for file in /dev/shm/*; do
   if [ -e "$file" ]; then
       echo "Removing $file..."
       rm "$file"
   fi
done

PROJ_DIR=/global/cfs/cdirs/m4645/alvinliu/repo/LLM-Shearing
LAUNCH_SCRIPT=/global/cfs/cdirs/m4645/alvinliu/repo/LLM-Shearing/llmshearing/scripts/launch.sh
DATA_DIR=/pscratch/sd/a/alvinliu/datasets/shearllm/for_prune
OUTPUT_DIR=/global/cfs/cdirs/m4645/alvinliu/workspace/results/shearllm
TRAIN_SCRIPT=${PROJ_DIR}/llmshearing/train.py
MODEL_PATH=${PROJ_DIR}/llmshearing/models/Llama-2-7b-composer

# Specify $PROJ_DIR in scripts/launch.sh and scripts/srun_launch.sh if using slurm

from_model=7b # source model size
config_file=${PROJ_DIR}/llmshearing/configs/cola-llama2/${from_model}-distill-hpc.yaml
path=$MODEL_PATH/state_dict.pt

# data setup
data_local=${DATA_DIR}

# basic setup
BZ=${BZ:-"2"}
TBZ=${TBZ:-"32"}
max_seq_len=4096
device_train_microbatch_size=$BZ
global_train_batch_size=$TBZ
device_eval_batch_size=8

# learning setup
lr=1e-3 # learning rate for the main parameters
max_duration=3200ba # 0.42B tokens
save_interval=100ba # save in the end
t_warmup=320ba # 10% learning rate warmup 

# dynamic loading setup
dynamic=True
set_names=[cc,github,book,stackexchange,wiki,arxiv,c4-rp] # domain names
proportion=[0.67,0.045,0.045,0.02,0.045,0.025,0.15] # initial proportion of RP, make sure that the sum(proportion) = 1
# doremi: update weights with exponential descent
# constant: keep the weights constant
update_type=constant 
target_loss=[1.7520,0.6216,1.9063,1.4482,1.4844,1.2637,1.9219]
eval_split_name=eval_merge # eval on all domains
eval_target_model=false # evaluate on the current model, not the target model, otherwise the loss will be inaccurate
eval_interval=50ba # eval every 50 batches and update the loading proportion


# pruning setup
# save directroy
TAG=${TAG:-"none"}
run_name=cola_llama2_${from_model}_distill_only_${update_type}_sl${max_seq_len}
if [ "${TAG}" != "none" ]; then
    run_name=${TAG}_${run_name}
fi
save_dir=${OUTPUT_DIR}/${run_name}
wandb_dir=${save_dir} # save locally

CONTINUE=${CONTINUE:-"none"}
if [ "${CONTINUE}" != "none" ]; then
    readonly load_flag="load_path=${CONTINUE} save_overwrite=true"
else
    readonly load_flag=""
fi

num_nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l)
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export MASTER_ADDR=$master_addr
export WORLD_SIZE=$(( $num_nodes * $SLURM_GPUS_PER_NODE ))
export MASTER_PORT=$(( 10000 + RANDOM % 10000 ))

echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE
echo "num_nodes="$num_nodes

export HF_HOME="/pscratch/sd/a/alvinliu/datasets/.cache/huggingface"

srun -u shifter torchrun --nproc-per-node=4 --master-port=$MASTER_PORT --nnodes=$SLURM_JOB_NUM_NODES \
    --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT $PROJ_DIR/llmshearing/train_torch.py \
    $config_file \
    run_name=${run_name} \
    data_local=${data_local} \
    eval_loader.dataset.split=${eval_split_name} \
    global_train_batch_size=${global_train_batch_size} \
    device_train_microbatch_size=${device_train_microbatch_size} \
    device_eval_batch_size=${device_eval_batch_size} \
    max_seq_len=${max_seq_len} \
    max_duration=${max_duration} \
    eval_first=false \
    scheduler.t_warmup=${t_warmup} \
    save_folder=${save_dir} \
    loggers.wandb.init_kwargs.dir=${wandb_dir} \
    eval_interval=${eval_interval} \
    save_interval=${save_interval} \
    optimizer.lr=${lr} \
    model.path=${path} \
    callbacks.data_loading.dynamic=${dynamic} \
    callbacks.data_loading.set_names=${set_names} \
    callbacks.data_loading.proportion=${proportion} \
    callbacks.data_loading.update_type=${update_type} \
    callbacks.data_loading.target_loss=${target_loss} \
    train_loader.num_workers=0 \
    train_loader.prefetch_factor=null \
    train_loader.persistent_workers=false \
    $load_flag