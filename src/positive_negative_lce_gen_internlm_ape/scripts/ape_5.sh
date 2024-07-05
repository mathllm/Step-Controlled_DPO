round=$1
address=$2

while true; do
    tmux new-session -d -s 5 "python src/positive_negative_lce_gen_internlm_ape/lce_solution_gen_ape.py $round -a $address -i 5"
    sleep 5
    
    while true; do
        sleep 5
        python src/positive_negative_lce_gen_internlm_ape/process_ape.py $round -i 5 && break
        sleep 30s
    done

    ((round++))

    sleep 5
done
