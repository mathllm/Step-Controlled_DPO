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
    parser.add_argument("d", help="dataset")
    args = parser.parse_args()
    return args

def watch(args):
    count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    dir = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/different_ranked_negative_divided_tmp1"
    while True:
        for i in range(3):
            file_paths = glob(f"{dir}/{args.d}/{i}_round*.jsonl")
            r = sorted([int(file_path.replace(f"{dir}/{args.d}/{i}_round", "").replace(".jsonl", "")) for file_path in file_paths])[-1]
            file_path = f"{dir}/{args.d}/{i}_round{r}.jsonl"
            if not os.path.exists(file_path):
                length = 0
            else:
                length = len(load_jsonl(file_path))
            source_file = f"{dir}/{args.d}/to_be_run_{i}_round{r}.jsonl"
            total_length = len(load_jsonl(source_file))
            if length == count[i]:
                print(f"round{r}: {i} no change: {length} ({total_length-length})")
            else:
                print(f"round{r}: {i}: {length} ({total_length-length})")
                count[i] = length


        print("\n***************************************\n")
        time.sleep(600)

if __name__ == "__main__":
    args = get_args()
    watch(args)