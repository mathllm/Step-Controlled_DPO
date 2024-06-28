round=$1


while true; do
    tmux new-session -d -s 0 "python /mnt/cache/luzimu/rlhf_math/src/positive_negative_lce_gen_1/lce_solution_gen_math.py $round -i 0"
    sleep 5
    
    while true; do
        sleep 5
        python /mnt/cache/luzimu/rlhf_math/src/positive_negative_lce_gen_1/process_math.py $round -i 0 && break
        sleep 30s
    done

    ((round++))

    sleep 5
done
