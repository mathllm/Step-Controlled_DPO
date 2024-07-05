round=$1
address=$2
index=$3

while true; do
    tmux new-session -d -s $index "python src/step_controled_dpo_lce_internlm/lce_solution_gen_math.py $round -a $address -i $index"
    sleep 5
    
    while true; do
        sleep 5
        python src/step_controled_dpo_lce_internlm/process_math.py $round -i $index && break
        sleep 30s
    done

    ((round++))

    sleep 5
done
