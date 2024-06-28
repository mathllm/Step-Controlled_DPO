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

def prepare_initial_data(in_file):
    datas = load_jsonl(in_file)
    seed(3407)
    shuffle(datas)
    new_datas = []

    for idx, data in tqdm(enumerate(datas)):
        extra = data["extra"]
        extra["question"] = data["question"]
        
        if len(data["correct_solutions"]) >= 1:
            correct_solution = data["correct_solutions"][0]
        else:
            correct_solution = []
            
        new_data = {
            "idx": idx,
            "extra": extra,
            "correct_solution": correct_solution,
            "wrong_solutions": []
        }
            
        new_datas.append(new_data)
        
    return new_datas

def is_noisy_data(debug_result):
    if len(debug_result) > 24:
        return True
    for block in debug_result:
        if len(block["content"]) > 3000 or "<|execution|>" in block["content"]:
            return True
    return False

def prepare_correct_incorrect_solutions(initial_file, in_files, out_file):
    new_datas = prepare_initial_data(initial_file)
    noisy_datas = []
    
    for in_file in tqdm(in_files):
        datas = load_jsonl(in_file)
        for data in tqdm(datas):
            if is_noisy_data(data["debug_result"]):
                noisy_datas.append(data)
                continue
            new_datas[data["idx"]]["wrong_solutions"].append(
                {
                    "start_steps": data["start_steps"],
                    "debug_result": data["debug_result"]
                }
            )
            
    save_jsonl(noisy_datas, out_file[:-6] + "_noisy.jsonl")
    save_jsonl(new_datas, out_file)
    
def get_messages_from_wrong_debug_result(debug_result, start_steps):
    messages = []
    messages.append({"role": "system", "content": [{"type": "text", "content": ""}]})
    messages.append({"role": "user", "content": [{"type": "text", "content": debug_result[1]["content"]}]})
    assistant_correct = []
    for block in debug_result[2:2+start_steps]:
        if block["role"] == "code":
            assistant_correct.append({
                "type": "code",
                "content": block["content"]
            },)
        elif block["role"] == "text":
            assistant_correct.append({
                "type": "text",
                "content": block["content"]
            },)
        elif block["role"] == "execution":
            assistant_correct.append({
                "type": "execution",
                "content": block["content"]
            },)
    messages.append({"role": "assistant_correct", "content": assistant_correct})
    assistant = []
    for block in debug_result[2:]:
        if block["role"] == "code":
            assistant.append({
                "type": "code",
                "content": block["content"]
            },)
        elif block["role"] == "text":
            assistant.append({
                "type": "text",
                "content": block["content"]
            },)
        elif block["role"] == "execution":
            assistant.append({
                "type": "execution",
                "content": block["content"]
            },)
    messages.append({"role": "assistant", "content": assistant})
    return messages

def main_gsm8k():
    initial_file = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mistral_lce_alignment_sample/gsm8k_train_lce_round53.jsonl"
    out_file = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/different_ranked_negative_divided/processed_results/gsm8k_train_lce_round53_step_controled_negative.jsonl"
    in_files = []
    for i in range(3):
        for j in range(100):
            in_file = f"/mnt/cache/luzimu/rlhf_math/data/lce_solutions/different_ranked_negative_divided/gsm8k/result_{i}_round{j}.jsonl"
            if os.path.isfile(in_file):
                in_files.append(in_file)
    prepare_correct_incorrect_solutions(initial_file, in_files, out_file)

def main_math():
    initial_file = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mistral_lce_alignment_sample/math_train_lce_round7.jsonl"
    out_file = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/different_ranked_negative_divided/processed_results/math_train_lce_round53_step_controled_negative.jsonl"
    in_files = []
    for i in range(3):
        for j in range(100):
            in_file = f"/mnt/cache/luzimu/rlhf_math/data/lce_solutions/different_ranked_negative_divided/math/result_{i}_round{j}.jsonl"
            if os.path.isfile(in_file):
                in_files.append(in_file)
    prepare_correct_incorrect_solutions(initial_file, in_files, out_file)
    
if __name__ == "__main__":
    main_gsm8k()
    main_math()