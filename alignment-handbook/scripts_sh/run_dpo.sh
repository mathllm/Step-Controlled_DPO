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

conda activate /mnt/cache/luzimu/rlhf_math/.env/handbookenv
cd /mnt/cache/luzimu/rlhf_math

export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0

wandb login "a8fe59167f5543baf6168a0cf5d52773a1bd6bf8"

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file /mnt/cache/luzimu/rlhf_math/alignment-handbook/recipes/accelerate_configs/deepspeed_zero3.yaml /mnt/cache/luzimu/rlhf_math/alignment-handbook/scripts/run_dpo.py /mnt/cache/luzimu/rlhf_math/alignment-handbook/recipes/llama-7b-ultrachat/dpo/config_full.yaml
