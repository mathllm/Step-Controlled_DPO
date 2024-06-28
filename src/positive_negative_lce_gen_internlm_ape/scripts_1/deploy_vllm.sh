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

conda activate /mnt/cache/luzimu/rlhf_math/.env/vllmenv

export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0

export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=13
export NCCL_IB_AR_THRESHOLD=0

export CUDA_VISIBLE_DEVICES=0,1,2,3

set -x

hostname -I # print the host ip

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
echo "Running on $CHUNKS GPUs: ${GPULIST[@]}"

# TODO: tensor-parallel-size > 1
# TODO: test the speed of 1 gpu and multiple gpus

python -m vllm.entrypoints.api_server \
--model /mnt/cache/luzimu/rlhf_math/alignment-handbook/outs/internlm2-20b_lce \
--trust-remote-code \
--seed 42 \
--tensor-parallel-size 1 \
--host 0.0.0.0 \
--port 8001