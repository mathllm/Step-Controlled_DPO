DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd /

conda init bash
apt-get update
apt-get install tmux -y
source ~/.bashrc

source /opt/conda/etc/profile.d/conda.sh
conda activate /mnt/cache/luzimu/rlhf_math/.env/inferenv

dataset=$1
round=$2
index=$3

address=$(hostname -I | awk '{print $1}')

tmux kill-server
tmux new-session -d -s loop${index} "bash /mnt/cache/luzimu/rlhf_math/src/positive_negative_lce_gen_mathcoder_mistral_addsys/scripts/${dataset}.sh ${round} ${address} ${index}"
tmux new-session -d -s watch "bash /mnt/cache/luzimu/rlhf_math/src/positive_negative_lce_gen_mathcoder_mistral_addsys/scripts/watch.sh ${index} ${dataset}"

sleep 3s
tmux ls