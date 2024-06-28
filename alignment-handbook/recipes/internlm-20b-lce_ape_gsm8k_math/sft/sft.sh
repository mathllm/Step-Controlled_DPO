__conda_setup="$('/usr/local/lib/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/local/lib/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/usr/local/lib/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/usr/local/lib/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

conda activate /mnt/cache/luzimu/rlhf_math/.env/handbookenv
cd /mnt/cache/luzimu/rlhf_math

export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0

wandb login ""

OMP_NUM_THREADS=1 torchrun --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT --nproc_per_node 4 /mnt/cache/luzimu/rlhf_math/alignment-handbook/scripts/run_sft_lce_internlm.py $DIR/config_full.yaml