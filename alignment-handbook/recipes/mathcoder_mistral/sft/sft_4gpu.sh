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

conda activate mathllm-finetune/.env/handbookenv
cd mathllm-finetune

export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0

export NCCL_IB_TIMEOUT=22   
export NCCL_IB_RETRY_CNT=13 
export NCCL_IB_AR_THRESHOLD=0

wandb login ""

CONFIG=${1}

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file open_source_repositories/rlhf_math/alignment-handbook/recipes/accelerate_configs/deepspeed_zero3_4gpu.yaml open_source_repositories/rlhf_math/alignment-handbook/scripts/run_sft_lce.py ${CONFIG}