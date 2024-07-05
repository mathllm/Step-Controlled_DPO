import json
import os
import re
from tqdm import tqdm
import time
from argparse import ArgumentParser

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

    for idx, data in tqdm(enumerate(datas)):
        extra = data["extra"]
        new_data = {
            "question": data["question"],
            "extra": extra,
            "correct_num": len(data["correct_solutions"]),
            "wrong_num": len(data["wrong_solutions"])
        }
        new_datas.append(new_data)

    total = len(new_datas)
    steps = (total + n - 1) // n
    for i in range(n):
        save_jsonl(new_datas[i * steps: i * steps + steps], os.path.join(out_dir, f"to_be_run_{i}_round1.jsonl"))
        save_jsonl(datas[i * steps: i * steps + steps], os.path.join(out_dir, f"result_{i}.jsonl"))

    
def main_gsm8k():
    in_file = "data/lce_solutions/mistral_lce_alignment_sample_1/gsm8k/result_0_round0.jsonl"
    out_dir = "data/lce_solutions/mistral_lce_alignment_sample_1/gsm8k"
    n = 6
    get_initial_data(in_file, out_dir, n)

def main_math():
    in_file = "data/lce_solutions/mistral_lce_alignment_sample_1/math/result_0_round0.jsonl"
    out_dir = "data/lce_solutions/mistral_lce_alignment_sample_1/math"
    n = 6
    get_initial_data(in_file, out_dir, n)

if __name__ == "__main__":
    main_gsm8k()
    main_math()