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


def watch():
    count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    rounds = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    dir = "data/lce_solutions/mathcoder_mistral_dpo_addsys/naive_dpo"
    
    while True:
        for i in range(16):
            file_paths = glob(f"{dir}/gsm8k/{i}_round*.jsonl")
            rs = sorted([int(file_path.replace(f"{dir}/gsm8k/{i}_round", "").replace(".jsonl", "")) for file_path in file_paths])
            if len(file_paths) > 0:
                r = rs[-1]
                file_path = f"{dir}/gsm8k/{i}_round{r}.jsonl"
                if not os.path.exists(file_path):
                    length = 0
                else:
                    length = len(load_jsonl(file_path))
                source_file = f"{dir}/gsm8k/to_be_run_{i}_round{r}.jsonl"
                total_length = len(load_jsonl(source_file))
                if length == count[i] and r == rounds[i]:
                    print(f"round{r}: {i} no change: {length} ({total_length-length})")
                else:
                    print(f"round{r}: {i}: {length} ({total_length-length})")
                    count[i] = length
                    rounds[i] = r
            else:
                source_file = f"{dir}/gsm8k/to_be_run_{i}_round1.jsonl"
                total_length = len(load_jsonl(source_file))
                print(f"round1: {i} no change: 0 ({total_length})")


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