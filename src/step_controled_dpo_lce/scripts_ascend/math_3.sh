round=$1


while true; do
    tmux new-session -d -s m_3 "python /mnt/cache/luzimu/rlhf_math/src/different_negative_gen/lce_solution_gen_different_negative_math_divided_ascend_loss.py $round -i 3"
    sleep 5
    
    while true; do
        sleep 5
        python /mnt/cache/luzimu/rlhf_math/src/different_negative_gen/process_finished_math.py $round -i 3 && break
        sleep 10m
    done

    ((round++))

    sleep 5
done
