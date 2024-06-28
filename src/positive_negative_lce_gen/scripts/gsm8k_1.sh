round=$1


while true; do
    tmux new-session -d -s 1 "python /mnt/cache/luzimu/rlhf_math/src/positive_negative_lce_gen_1/lce_solution_gen_gsm8k.py $round -i 1"
    sleep 5
    
    while true; do
        sleep 5
        python /mnt/cache/luzimu/rlhf_math/src/positive_negative_lce_gen_1/process_gsm8k.py $round -i 1 && break
        sleep 30s
    done

    ((round++))

    sleep 5
done
