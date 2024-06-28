import json
import os
import re
from latex2sympy2 import latex2sympy
from tqdm import tqdm
from argparse import ArgumentParser

def save_jsonl(datas, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        for data in datas:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

def load_jsonl(in_file):
    with open(in_file, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]
    return datas

def main():
    in_file = "/mnt/cache/luzimu/code_generation-master/data/train/back_translation/train_solver/with_sys_gsm8k_math_81087.jsonl"
    datas = load_jsonl(in_file)
    gsm8k_file = ""
    gsm8k_datas = load_jsonl("/mnt/cache/luzimu/datasets_en/GSM8K/GSM8K_train.jsonl")
    gsm8k_questions = [e["question"] for e in gsm8k_datas]
    gsm8k_num = 0
    for data in tqdm(datas):
        if data["messages"][1]["content"][0]["content"] in gsm8k_questions:
            gsm8k_num += 1

    total_num = len(datas)
    print(f"gsm8k: {gsm8k_num}")
    print(f"math: {total_num - gsm8k_num}")
    print(f"total: {total_num}")
    
def main():
    in_file = "/mnt/cache/luzimu/code_generation-master/data/train/back_translation/train_solver/with_sys_gsm8k_math_81087.jsonl"
    datas = load_jsonl(in_file)
    gsm8k_file = ""
    gsm8k_datas = load_jsonl("/mnt/cache/luzimu/datasets_en/GSM8K/GSM8K_train.jsonl")
    gsm8k_questions = [e["question"] for e in gsm8k_datas]
    gsm8k_num = 0
    for data in tqdm(datas):
        if data["messages"][1]["content"][0]["content"] in gsm8k_questions:
            gsm8k_num += 1

    total_num = len(datas)
    print(f"gsm8k: {gsm8k_num}")
    print(f"math: {total_num - gsm8k_num}")
    print(f"total: {total_num}")

def main1():
    in_file = "/mnt/cache/luzimu/rlhf_math/data/internlm_gsm8k_math_no_ch_1x1_1x3/data/train/math_gsm8k_train.jsonl"
    datas = load_jsonl(in_file)
    gsm8k_file = ""
    gsm8k_datas = load_jsonl("/mnt/cache/luzimu/datasets_en/GSM8K/GSM8K_train.jsonl")
    math_datas = load_jsonl("/mnt/cache/luzimu/datasets_en/MATH/MATH_train.jsonl")
    gsm8k_questions = [e["question"] for e in gsm8k_datas]
    math_questions = [e["question"] for e in math_datas]
    gsm8k_num = 0
    math_num = 0
    for data in tqdm(datas):
        if data["chosen"][1]["content"][0]["content"] in gsm8k_questions:
            gsm8k_num += 1
        elif data["chosen"][1]["content"][0]["content"] in math_questions:
            math_num += 1

    total_num = len(datas)
    print(f"gsm8k: {gsm8k_num}")
    print(f"math: {math_num}")
    print(f"ape: {total_num - math_num - ape_num}")
    print(f"total: {total_num}")

if __name__ == "__main__":
    main1()