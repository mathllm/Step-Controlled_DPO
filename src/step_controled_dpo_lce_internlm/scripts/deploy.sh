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

tmux kill-server

dataset=$1
round=$2
index=$3

address=$(hostname -I | awk '{print $1}')

conda activate cloud-ai-lab
pip install latex2sympy2
pip install Levenshtein

tmux new-session -d -s loop${index} "bash src/step_controled_dpo_lce_internlm/scripts/${dataset}.sh ${round} ${address} ${index}"
tmux new-session -d -s watch "bash src/step_controled_dpo_lce_internlm/scripts/watch.sh ${index} ${dataset}"

sleep 2s
tmux ls
