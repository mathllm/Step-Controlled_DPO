round=$1


while true; do
    tmux new-session -d -s g_3 "python src/different_negative_gen/lce_solution_gen_different_negative_gsm8k_divided_ascend_loss.py $round -i 3"
    sleep 5
    
    while true; do
        sleep 5
        python src/different_negative_gen/process_finished_gsm8k.py $round -i 3 && break
        sleep 10m
    done

    ((round++))

    sleep 5
done
