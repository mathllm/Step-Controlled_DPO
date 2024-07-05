import json
import os
import re
from tqdm import tqdm
import time
from argparse import ArgumentParser
from random import shuffle, seed

def save_jsonl(datas, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        for data in datas:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

def load_jsonl(in_file):
    with open(in_file, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]
    return datas

def process_metamath_to_alignment(in_file, out_file_train, out_file_test):
    datas = load_jsonl(in_file)
    new_datas = []
    for data in tqdm(datas):
        new_data = {
            "messages": [
                {"role": "user", "content": data["query"]},
                {"role": "assistant", "content": data["response"]}
            ]
        }
        new_datas.append(new_data)

    seed(3407)
    shuffle(new_datas)
    
    test_train_split = int(0.01 * len(new_datas))
    save_jsonl(new_datas[:test_train_split], out_file_test)
    save_jsonl(new_datas[test_train_split:], out_file_train)

def main():
    in_file = "data/MetaMathQA_processed/MetaMathQA-395K.jsonl"
    out_file_train = "data/MetaMathQA_alignment/train/train.jsonl"
    out_file_test = "data/MetaMathQA_alignment/test/test.jsonl"
    process_metamath_to_alignment(in_file, out_file_train, out_file_test)

if __name__ == "__main__":
    main()