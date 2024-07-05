import json
import os
import re
from tqdm import tqdm
import time
from argparse import ArgumentParser
from random import seed, shuffle

def save_jsonl(datas, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        for data in datas:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

def load_jsonl(in_file):
    with open(in_file, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]
    return datas

def prepare_initial_data(in_file, out_dir):
    datas = load_jsonl(in_file)
    seed(3407)
    shuffle(datas)
    new_datas = []

    for data in tqdm(datas):
        extra = data["extra"]
        extra["question"] = data["question"]
        if len(data["correct_solutions"]) >= 1:
            new_data = {
                "extra": extra,
                "correct_solution": data["correct_solutions"][0]
            }
        else:
            new_data = {
                "extra": extra,
            }
        new_datas.append(new_data)

    step = 2500
    n = (len(datas) + step - 1) // step
    for i in range(n):
        save_jsonl(new_datas[i * step: i * step + step], os.path.join(out_dir, f"{i}_round0.jsonl"))

def prepare_initial_data_divided(in_file, out_dir):
    datas = load_jsonl(in_file)
    seed(3407)
    shuffle(datas)
    new_datas = []

    for idx, data in tqdm(enumerate(datas)):
        extra = data["extra"]
        extra["question"] = data["question"]
        
        # enter step=0
        messages_to_be_run = [
            {"role": "system", "content": ""},
            {"role": "user", "content": data["question"]}
        ]
        new_data = {
            "idx": idx,
            "start_steps": 0,
            "extra": extra,
            "messages_to_be_run": messages_to_be_run
        }
        new_datas.append(new_data)
        
        # enter various steps
        if len(data["correct_solutions"]) >= 1:
            correct_solution = data["correct_solutions"][0]
            for i in range(2, len(correct_solution) - 2):
                if correct_solution[i]["role"] == "text" or correct_solution[i]["role"] == "execution":
                    messages_to_be_run = correct_solution[:i + 1]
                    new_data = {
                        "idx": idx,
                        "start_steps": i - 1,
                        "extra": extra,
                        "messages_to_be_run": messages_to_be_run
                    }
                    new_datas.append(new_data)
                    
    print(f"length: {len(new_datas)}")
    n = 6
    step = (len(new_datas) + n - 1) // n
    for i in range(n):
        save_jsonl(new_datas[i * step: i * step + step], os.path.join(out_dir, f"to_be_run_{i}_round1.jsonl"))


def main_gsm8k():
    in_file = "data/lce_solutions/mistral_lce_alignment_sample/gsm8k_train_lce_round53.jsonl"
    out_dir = "data/lce_solutions/different_ranked_negative_divided/gsm8k"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    prepare_initial_data_divided(in_file, out_dir)

def main_math():
    in_file = "data/lce_solutions/mistral_lce_alignment_sample/math_train_lce_round7.jsonl"
    out_dir = "data/lce_solutions/different_ranked_negative_divided/math"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    prepare_initial_data_divided(in_file, out_dir)

def main_gsm8k_ascend():
    in_file = "data/lce_solutions/mistral_lce_alignment_sample/gsm8k_train_lce_round53.jsonl"
    out_dir = "data/lce_solutions/ascending_temperature_negative/gsm8k"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    prepare_initial_data_divided(in_file, out_dir)

def main_math_ascend():
    in_file = "data/lce_solutions/mistral_lce_alignment_sample/math_train_lce_round7.jsonl"
    out_dir = "data/lce_solutions/ascending_temperature_negative/math"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    prepare_initial_data_divided(in_file, out_dir)
    
if __name__ == "__main__":
    main_gsm8k_ascend()
    main_math_ascend()