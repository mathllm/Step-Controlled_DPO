import json
import os
import re
from tqdm import tqdm
import time
from argparse import ArgumentParser

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
    parser.add_argument("r", type=int, help="round number")
    parser.add_argument("-d", help="dataset")
    args = parser.parse_args()
    return args

def watch(args):
    count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    while True:
        for i in range(3):
            file_path = f"data/lce_solutions/different_ranked_negative_divided_tmp1/{args.d}/{i}_round{args.r}.jsonl"
            if not os.path.exists(file_path):
                length = 0
            else:
                length = len(load_jsonl(file_path))
            source_file = f"data/lce_solutions/different_ranked_negative_divided_tmp1/{args.d}/to_be_run_{i}_round{args.r}.jsonl"
            total_length = len(load_jsonl(source_file))
            if length == count[i]:
                print(f"{i} no change: {length} ({total_length-length})")
            else:
                print(f"{i}: {length} ({total_length-length})")
                count[i] = length


        print("\n***************************************\n")
        time.sleep(600)

if __name__ == "__main__":
    args = get_args()
    watch(args)