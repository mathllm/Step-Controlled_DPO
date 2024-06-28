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

def prepare_initial_data_divided(in_file, out_dir, n=6):
    datas = load_jsonl(in_file)
    seed(3407)
    shuffle(datas)
    new_datas = []

    for idx, data in tqdm(enumerate(datas)):
        if "extra" in data:
            extra = data["extra"]
            extra["question"] = data["question"]
        else:
            extra = {
                "id": data["id"],
                "question": data["question"],
                "answer": data["answer"],
                "equation": data["equation"],
                "idx": data["idx"]
            }
        
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
    step = (len(new_datas) + n - 1) // n
    for i in range(n):
        save_jsonl(new_datas[i * step: i * step + step], os.path.join(out_dir, f"to_be_run_{i}_round1.jsonl"))


def prepare_initial_data_divided_over1(in_file, out_dir, n=6):
    datas = load_jsonl(in_file)
    seed(3407)
    shuffle(datas)
    new_datas = []

    for idx, data in tqdm(enumerate(datas)):
        if "extra" in data:
            extra = data["extra"]
            extra["question"] = data["question"]
        else:
            extra = {
                "id": data["id"],
                "question": data["question"],
                "answer": data["answer"],
                "equation": data["equation"],
                "idx": data["idx"]
            }
        
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
    step = (len(new_datas) + n - 1) // n
    for i in range(n):
        save_jsonl(new_datas[i * step: i * step + step], os.path.join(out_dir, f"to_be_run_{i}_round1.jsonl"))


def main_gsm8k():
    in_file = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mistral_lce_alignment_sample/gsm8k_train_lce_round53.jsonl"
    out_dir = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/different_ranked_negative_divided/gsm8k"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    prepare_initial_data_divided(in_file, out_dir)

def main_math():
    in_file = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mistral_lce_alignment_sample/math_train_lce_round7.jsonl"
    out_dir = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/different_ranked_negative_divided/math"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    prepare_initial_data_divided(in_file, out_dir)

def main_gsm8k_ascend():
    in_file = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mistral_lce_alignment_sample/gsm8k_train_lce_round53.jsonl"
    out_dir = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/ascending_temperature_negative/gsm8k"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    prepare_initial_data_divided(in_file, out_dir)

def main_math_ascend():
    in_file = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mistral_lce_alignment_sample/math_train_lce_round7.jsonl"
    out_dir = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/ascending_temperature_negative/math"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    prepare_initial_data_divided(in_file, out_dir)

def main_gsm8k_ascend_internlm():
    in_file = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/gsm8k_results_7473.jsonl"
    out_dir = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/sc_dpo/gsm8k"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    n = 12
    prepare_initial_data_divided(in_file, out_dir, n)

def main_math_ascend_internlm():
    in_file = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/math_results_7500.jsonl"
    out_dir = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/sc_dpo/math"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    n = 12
    prepare_initial_data_divided(in_file, out_dir, n)

def main_ape_ascend_internlm():
    in_file = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/ape_results_200488.jsonl"
    out_dir = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/sc_dpo/ape"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    n = 12
    prepare_initial_data_divided(in_file, out_dir, n)
    
def main_gsm8k_ascend_internlm_over1():
    in_file = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/gsm8k_results_7473.jsonl"
    out_dir = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/sc_dpo/gsm8k"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    n = 18
    prepare_initial_data_divided_over1(in_file, out_dir, n)
    
def main_math_ascend_internlm_over1():
    in_file = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/math_results_7500.jsonl"
    out_dir = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/sc_dpo/math"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    n = 18
    prepare_initial_data_divided_over1(in_file, out_dir, n)

def main_ape_ascend_internlm_over1():
    in_file = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/ape_results_200488.jsonl"
    out_dir = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/sc_dpo/ape"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    n = 18
    prepare_initial_data_divided_over1(in_file, out_dir, n)
    
def main_gsm8k_ascend_internlm_over1_no_ch():
    in_file = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_no_ch_negative_positive_gen/naive_dpo/processed_results/gsm8k_results_7473.jsonl"
    out_dir = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_no_ch_negative_positive_gen/sc_dpo/gsm8k"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    n = 18
    prepare_initial_data_divided_over1(in_file, out_dir, n)
    
def main_math_ascend_internlm_over1_no_ch():
    in_file = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_no_ch_negative_positive_gen/naive_dpo/processed_results/math_results_7500.jsonl"
    out_dir = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_no_ch_negative_positive_gen/sc_dpo/math"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    n = 18
    prepare_initial_data_divided_over1(in_file, out_dir, n)
    
    
def main_gsm8k_mathcoder():
    in_file = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mathcoder_dpo/naive_dpo/processed_results/gsm8k_results_7473.jsonl"
    out_dir = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mathcoder_dpo/sc_dpo/gsm8k"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    n = 10
    prepare_initial_data_divided_over1(in_file, out_dir, n)
    
def main_math_mathcoder():
    in_file = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mathcoder_dpo/naive_dpo/processed_results/math_results_7500.jsonl"
    out_dir = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mathcoder_dpo/sc_dpo/math"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    n = 10
    prepare_initial_data_divided_over1(in_file, out_dir, n)
    
def main_gsm8k_direct_error():
    in_file = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mistral_lce_alignment_sample/gsm8k_train_lce_round53.jsonl"
    out_dir = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/direct_error_creation/create_error/gsm8k"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    prepare_initial_data_divided_over1(in_file, out_dir, n=1)

def main_math_direct_error():
    in_file = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mistral_lce_alignment_sample/math_train_lce_round7.jsonl"
    out_dir = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/direct_error_creation/create_error/math"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    prepare_initial_data_divided_over1(in_file, out_dir, n=1)
    
if __name__ == "__main__":
    main_gsm8k_direct_error()
    main_math_direct_error()