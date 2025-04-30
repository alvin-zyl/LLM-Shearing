PROJ_DIR=/global/cfs/cdirs/m4645/alvinliu/repo/LLM-Shearing

echo ${SLURM_NODEID} 
composer --node_rank ${SLURM_NODEID} $PROJ_DIR/llmshearing/train.py $@  