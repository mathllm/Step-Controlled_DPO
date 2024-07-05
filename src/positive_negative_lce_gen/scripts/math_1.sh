round=$1


while true; do
    tmux new-session -d -s 1 "python src/positive_negative_lce_gen_1/lce_solution_gen_math.py $round -i 1"
    sleep 5
    
    while true; do
        sleep 5
        python src/positive_negative_lce_gen_1/process_math.py $round -i 1 && break
        sleep 30s
    done

    ((round++))

    sleep 5
done
