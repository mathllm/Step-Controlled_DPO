round=$1
address=$2
index=$3

while true; do
    tmux new-session -d -s $index "python /mnt/cache/luzimu/rlhf_math/src/positive_negative_lce_gen_internlm_no_ch/lce_solution_gen_gsm8k.py $round -a $address -i $index"
    sleep 5
    
    while true; do
        sleep 5
        python /mnt/cache/luzimu/rlhf_math/src/positive_negative_lce_gen_internlm_no_ch/process_gsm8k.py $round -i $index && break
        sleep 30s
    done

    ((round++))

    sleep 5
done
