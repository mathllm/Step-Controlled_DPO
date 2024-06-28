round=$1


while true; do
    tmux new-session -d -s g_1 "python /mnt/cache/luzimu/rlhf_math/src/different_negative_gen/lce_solution_gen_different_negative_gsm8k_divided_ascend_loss.py $round -i 1"
    sleep 5
    
    while true; do
        sleep 5
        python /mnt/cache/luzimu/rlhf_math/src/different_negative_gen/process_finished_gsm8k.py $round -i 1 && break
        sleep 10m
    done

    ((round++))

    sleep 5
done
