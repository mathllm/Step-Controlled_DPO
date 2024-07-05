import json
import os
import re
from tqdm import tqdm
import time
from argparse import ArgumentParser

from random import seed, shuffle

def save_jsonl(datas, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        for idx, data in enumerate(datas):
            data["idx"] = idx
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

def load_jsonl(in_file):
    with open(in_file, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]
    return datas

def get_initial_data(in_file, out_dir, n):
    datas = load_jsonl(in_file)
    new_datas = []
    
    seed(3407)
    shuffle(datas)

    for idx, data in tqdm(enumerate(datas)):
        if "extra" in data.keys():
            extra = data["extra"]
        else:
            extra = data
        correct_num = 0
        wrong_num = 0
        if "correct_solutions" in data.keys():
            correct_num = len(data["correct_solutions"])
        if "wrong_solutions" in data.keys():
            wrong_num = len(data["wrong_solutions"])
        new_data = {
            "question": data["question"],
            "extra": extra,
            "correct_num": correct_num,
            "wrong_num": wrong_num
        }
        new_datas.append(new_data)

    total = len(new_datas)
    steps = (total + n - 1) // n
    for i in range(n):
        save_jsonl(new_datas[i * steps: i * steps + steps], os.path.join(out_dir, f"to_be_run_{i}_round1.jsonl"))
        save_jsonl(datas[i * steps: i * steps + steps], os.path.join(out_dir, f"result_{i}.jsonl"))


def get_initial_data_milti_infiles(in_files, out_dir, n):
    datas = []
    for in_file in in_files:
        datas.extend(load_jsonl(in_file))

    seed(3407)
    shuffle(datas)

    new_datas = []

    for idx, data in tqdm(enumerate(datas)):
        if "extra" in data.keys():
            extra = data["extra"]
        else:
            extra = {k: v for k, v in data.items() if not k.endswith("solutions")}
        
        if "correct_solutions" not in data.keys():
            data["correct_solutions"] = []
        if "wrong_solutions" not in data.keys():
            data["wrong_solutions"] = []
        if "correct_errored_solutions" not in data.keys():
            data["correct_errored_solutions"] = []

        correct_num = len(data["correct_solutions"])
        wrong_num = len(data["wrong_solutions"])
        new_data = {
            "question": data["question"],
            "extra": extra,
            "correct_num": correct_num,
            "wrong_num": wrong_num
        }
        new_datas.append(new_data)

    total = len(new_datas)
    steps = (total + n - 1) // n
    for i in range(n):
        save_jsonl(new_datas[i * steps: i * steps + steps], os.path.join(out_dir, f"to_be_run_{i}_round1.jsonl"))
        save_jsonl(datas[i * steps: i * steps + steps], os.path.join(out_dir, f"result_{i}.jsonl"))


def main_gsm8k():
    in_file = "datasets_en/GSM8K/GSM8K_train.jsonl"
    out_dir = "data/lce_solutions/mathcoder_mistral_dpo_addsys/naive_dpo/gsm8k"
    n = 7
    get_initial_data(in_file, out_dir, n)

def main_math():
    in_file = "datasets_en/MATH/MATH_train.jsonl"
    out_dir = "data/lce_solutions/mathcoder_mistral_dpo_addsys/naive_dpo/math"
    n = 7
    get_initial_data(in_file, out_dir, n)

def main_math_multi():
    in_files = [f"data/lce_solutions/mathcoder_mistral_dpo_addsys/naive_dpo/math1/result_{i}.jsonl" for i in range(7)]
    out_dir = "data/lce_solutions/mathcoder_mistral_dpo_addsys/naive_dpo/math"
    n = 16
    get_initial_data_milti_infiles(in_files, out_dir, n)
    
def main_gsm8k_multi():
    in_files = [f"data/lce_solutions/mathcoder_mistral_dpo_addsys/naive_dpo/gsm8k1/result_{i}.jsonl" for i in range(7)]
    out_dir = "data/lce_solutions/mathcoder_mistral_dpo_addsys/naive_dpo/gsm8k"
    n = 16
    get_initial_data_milti_infiles(in_files, out_dir, n)
    

if __name__ == "__main__":
    main_math_multi()
    main_gsm8k_multi()