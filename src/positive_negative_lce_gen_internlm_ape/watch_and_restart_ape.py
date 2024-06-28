import json
import os
import re
from tqdm import tqdm
import time
from argparse import ArgumentParser
from glob import glob

def save_jsonl(datas, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        for data in datas:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

def load_jsonl(in_file):
    with open(in_file, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]
    return datas

def get_args():
    parser = ArgumentParser(description="A simple argument parser")
    parser.add_argument("i", type=int, help="round number")
    args = parser.parse_args()
    return args

def restart_session(r, i):
    os.system(f"tmux kill-session -t {i}")
    cmd = f"tmux new-session -d -s {i} 'python /mnt/cache/luzimu/rlhf_math/src/positive_negative_lce_gen_internlm_ape/lce_solution_gen_ape.py {r} -i {i}'"
    os.system(cmd)

def restart_loop(r, i):
    os.system(f"tmux kill-session -t loop{i}")
    cmd = f"tmux new-session -d -s loop{i} 'bash /mnt/cache/luzimu/rlhf_math/src/positive_negative_lce_gen_internlm_ape/scripts/ape_{i}.sh {r}'"
    os.system(cmd)

def watch():
    count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    dir = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/naive_dpo"
    args = get_args()
    i = args.i

    file_paths = glob(f"{dir}/ape/{i}_round*.jsonl")
    r = sorted([int(file_path.replace(f"{dir}/ape/{i}_round", "").replace(".jsonl", "")) for file_path in file_paths])[-1]
    file_path = f"{dir}/ape/{i}_round{r}.jsonl"
    if not os.path.exists(file_path):
        length = 0
    else:
        length = len(load_jsonl(file_path))
    source_file = f"{dir}/ape/to_be_run_{i}_round{r}.jsonl"
    total_length = len(load_jsonl(source_file))
    if length == count[i]:
        print(f"round{r}: {i} no change: {length} ({total_length-length})")
        if length != total_length:
            restart_session(r, i)
        elif os.path.exists(f"{dir}/ape/to_be_run_{i}_round{r + 1}.jsonl"):
            restart_session(r + 1, i)
        else:
            restart_loop(r, i)
    else:
        print(f"round{r}: {i}: {length} ({total_length-length})")
        count[i] = length


    print("\n***************************************\n")
    time.sleep(600)

def main_test():
    os.system("tmux kill-session -t loop")
    tmux_cmd = "'sleep inf'"
    os.system(f"tmux new-session -d -s loop1 {tmux_cmd}")

def main_test_conda():
    os.system("conda info")

def main():
    watch()

if __name__ == "__main__":
    main()