index=$1
dataset=$2


while true; do
    python src/step_controled_dpo_lce_internlm/watch_and_restart_${dataset}.py ${index}

    sleep 5
done