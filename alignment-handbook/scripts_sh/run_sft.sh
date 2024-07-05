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

conda activate .env/handbookenv
cd rlhf_math

export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0

wandb login ""

config=$1

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file alignment-handbook/recipes/accelerate_configs/deepspeed_zero3.yaml alignment-handbook/scripts/run_sft.py $config
