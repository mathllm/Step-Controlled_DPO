index=$1
dataset=$2


while true; do
    python src/positive_negative_lce_gen_internlm_ape/watch_and_restart_${dataset}.py ${index}

    sleep 5
done