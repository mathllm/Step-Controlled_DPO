round=$1
address=$2
index=$3

while true; do
    tmux new-session -d -s $index "python src/positive_negative_lce_gen_internlm_ape/lce_solution_gen_math_1.py $round -a $address -i $index"
    sleep 5
    
    while true; do
        sleep 5
        python src/positive_negative_lce_gen_internlm_ape/process_math_1.py $round -i $index && break
        sleep 30s
    done

    ((round++))

    sleep 5
done
