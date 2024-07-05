round=$1
address=$2

while true; do
    tmux new-session -d -s 9 "python src/positive_negative_lce_gen_internlm_ape/lce_solution_gen_gsm8k.py $round -a $address -i 9"
    sleep 5
    
    while true; do
        sleep 5
        python src/positive_negative_lce_gen_internlm_ape/process_gsm8k.py $round -i 9 && break
        sleep 30s
    done

    ((round++))

    sleep 5
done
