round=$1
address=$2

while true; do
    tmux new-session -d -s 3 "python /mnt/cache/luzimu/rlhf_math/src/positive_negative_lce_gen_internlm_ape/lce_solution_gen_math.py $round -a $address -i 3"
    sleep 5
    
    while true; do
        sleep 5
        python /mnt/cache/luzimu/rlhf_math/src/positive_negative_lce_gen_internlm_ape/process_math.py $round -i 3 && break
        sleep 30s
    done

    ((round++))

    sleep 5
done
