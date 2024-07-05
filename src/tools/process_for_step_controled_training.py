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
        if "extra" in data:
            extra = data["extra"]
        else:
            extra = {k: v for k, v in data.items() if k != "correct_solutions" and k != "wrong_solutions"}
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
        for data in datas:
            if is_noisy_data(data["debug_result"]):
                noisy_datas.append(data)
                continue
            new_datas[data["idx"]]["wrong_solutions"].append(
                {
                    "start_steps": data["start_steps"],
                    "debug_result": data["debug_result"]
                }
            )
            
    print(len(new_datas))
    save_jsonl(noisy_datas, out_file[:-6] + "_noisy.jsonl")
    save_jsonl(new_datas, out_file)
    
def get_messages_from_debug_result(debug_result, start_steps):
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
    for block in debug_result[2+start_steps:]:
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

def get_messages_from_debug_result_naive(debug_result):
    messages = []
    messages.append({"role": "system", "content": [{"type": "text", "content": ""}]})
    messages.append({"role": "user", "content": [{"type": "text", "content": debug_result[1]["content"]}]})
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

def get_chosen_rejected_controled_steps(in_files, out_train_file, out_test_file):
    new_datas = []
    for in_file in in_files:
        datas = load_jsonl(in_file)
        for data in tqdm(datas):
            if len(data["correct_solution"]) > 0 and len(data["wrong_solutions"]) > 0:
                chosen = get_messages_from_debug_result(data["correct_solution"], 0)
                for wrong_solution in data["wrong_solutions"]:
                    new_data = {
                        "chosen": get_messages_from_debug_result(data["correct_solution"], wrong_solution["start_steps"]),
                        "rejected": get_messages_from_debug_result(wrong_solution["debug_result"], wrong_solution["start_steps"])
                    }
                    new_datas.append(new_data)
        print(f"{len(new_datas)}\n")
    seed(3407)
    shuffle(new_datas)
    split_idx = int(len(new_datas) * 0.01)
    save_jsonl(new_datas[:split_idx], out_test_file)
    save_jsonl(new_datas[split_idx:], out_train_file)
    
def get_chosen_rejected_controled_steps_limited(in_files, out_train_file, out_test_file):
    new_datas = []
    for in_file, neg_limit in in_files:
        datas = load_jsonl(in_file)
        for data in tqdm(datas):
            if len(data["correct_solution"]) > 0 and len(data["wrong_solutions"]) > 0:
                chosen = get_messages_from_debug_result(data["correct_solution"], 0)
                for wrong_solution in data["wrong_solutions"][:neg_limit]:
                    new_data = {
                        "chosen": get_messages_from_debug_result(data["correct_solution"], wrong_solution["start_steps"]),
                        "rejected": get_messages_from_debug_result(wrong_solution["debug_result"], wrong_solution["start_steps"])
                    }
                    new_datas.append(new_data)
        print(f"{len(new_datas)}\n")
    seed(3407)
    shuffle(new_datas)
    split_idx = int(len(new_datas) * 0.01)
    save_jsonl(new_datas[:split_idx], out_test_file)
    save_jsonl(new_datas[split_idx:], out_train_file)
    
def get_chosen_rejected_controled_steps_limited_add_dpo1x1(in_files, in_files_and_num, out_train_file, out_test_file, weights=(1, 1, 1), weights_sc=(1, 1, 1)):
    new_datas = []
    idx = 0
    pre_num = 0
    for in_file, neg_limit in in_files:
        datas = load_jsonl(in_file)
        datas = datas[: int(weights_sc[idx] * len(datas))]
        idx += 1
        for data in tqdm(datas):
            if len(data["correct_solution"]) > 0 and len(data["wrong_solutions"]) > 0:
                chosen = get_messages_from_debug_result(data["correct_solution"], 0)
                for wrong_solution in data["wrong_solutions"]:
                    if wrong_solution["start_steps"] == 0:
                        continue
                    neg_limit -= 1
                    new_data = {
                        "chosen": get_messages_from_debug_result(data["correct_solution"], wrong_solution["start_steps"]),
                        "rejected": get_messages_from_debug_result(wrong_solution["debug_result"], wrong_solution["start_steps"])
                    }
                    new_datas.append(new_data)
                    if neg_limit == 0:
                        break
        print(f"{len(new_datas) - pre_num}\n")
        pre_num = len(new_datas)
    new_datas = new_datas + get_chosen_rejected_alignment_lce_multi_file_diff(in_files_and_num, weights)
    seed(3407)
    shuffle(new_datas)
    split_idx = int(len(new_datas) * 0.01)
    save_jsonl(new_datas[:split_idx], out_test_file)
    save_jsonl(new_datas[split_idx:], out_train_file)
    
def get_chosen_rejected_controled_steps_only_later_steps(in_files, out_train_file, out_test_file):
    new_datas = []
    for in_file in in_files:
        datas = load_jsonl(in_file)
        for data in tqdm(datas):
            if len(data["correct_solution"]) > 0 and len(data["wrong_solutions"]) > 0:
                chosen = get_messages_from_debug_result(data["correct_solution"], 0)
                for wrong_solution in data["wrong_solutions"]:
                    if wrong_solution["start_steps"] > 0:
                        new_data = {
                            "chosen": get_messages_from_debug_result(data["correct_solution"], wrong_solution["start_steps"]),
                            "rejected": get_messages_from_debug_result(wrong_solution["debug_result"], wrong_solution["start_steps"])
                        }
                        new_datas.append(new_data)
        print(f"{len(new_datas)}\n")
    seed(3407)
    shuffle(new_datas)
    split_idx = int(len(new_datas) * 0.01)
    save_jsonl(new_datas[:split_idx], out_test_file)
    save_jsonl(new_datas[split_idx:], out_train_file)

def get_chosen_rejected_controled_steps_naive(in_files, out_train_file, out_test_file):
    new_datas = []
    for in_file in in_files:
        datas = load_jsonl(in_file)
        for data in tqdm(datas):
            if len(data["correct_solution"]) > 0 and len(data["wrong_solutions"]) > 0:
                chosen = get_messages_from_debug_result(data["correct_solution"], 0)
                for wrong_solution in data["wrong_solutions"]:
                    new_data = {
                        "chosen": get_messages_from_debug_result_naive(data["correct_solution"]),
                        "rejected": get_messages_from_debug_result_naive(wrong_solution["debug_result"])
                    }
                    new_datas.append(new_data)
        print(f"{len(new_datas)}\n")
    seed(3407)
    shuffle(new_datas)
    split_idx = int(len(new_datas) * 0.01)
    save_jsonl(new_datas[:split_idx], out_test_file)
    save_jsonl(new_datas[split_idx:], out_train_file)
    
def get_chosen_rejected_controled_steps_test_template(in_files, out_train_file, out_test_file):
    new_datas = []
    for in_file in in_files:
        datas = load_jsonl(in_file)
        for data in tqdm(datas):
            if len(data["correct_solution"]) > 0 and len(data["wrong_solutions"]) > 0:
                for wrong_solution in data["wrong_solutions"]:
                    if wrong_solution["start_steps"] > 0:
                        new_data = {
                            "chosen": get_messages_from_debug_result(data["correct_solution"], wrong_solution["start_steps"]),
                            "rejected": get_messages_from_debug_result(wrong_solution["debug_result"], wrong_solution["start_steps"])
                        }
                        new_datas.append(new_data)
        print(f"{len(new_datas)}\n")
    seed(3407)
    shuffle(new_datas)
    split_idx = int(len(new_datas) * 0.01)
    save_jsonl(new_datas[:split_idx], out_test_file)
    save_jsonl(new_datas[split_idx:], out_train_file)
    
def get_chosen_rejected_alignment_lce_multi_file_diff(in_files_and_num, weight=(1, 1, 1)):
    new_datas = []
    idx = 0
    pre_num = 0
    for in_file, num_correct, num_wrong in in_files_and_num:
        datas = load_jsonl(in_file)
        datas = datas[:int(weight[idx] * len(datas))]
        idx += 1
        for data in tqdm(datas):
            for correct_solution in data["correct_solutions"][:num_correct]:
                for wrong_solution in data["wrong_solutions"][:num_wrong]:
                    new_data = {
                        "chosen": get_messages_from_debug_result(correct_solution, 0),
                        "rejected": get_messages_from_debug_result(wrong_solution, 0)
                    }
                    new_datas.append(new_data)

        print(len(new_datas) - pre_num)
        pre_num = len(new_datas)
    return new_datas

def get_chosen_rejected_controled_steps_add_1x3_1x3(in_files, in_files_and_num, out_train_file, out_test_file):
    new_datas = []
    for in_file in in_files:
        datas = load_jsonl(in_file)
        for data in tqdm(datas):
            if len(data["correct_solution"]) > 0 and len(data["wrong_solutions"]) > 0:
                chosen = get_messages_from_debug_result(data["correct_solution"], 0)
                for wrong_solution in data["wrong_solutions"]:
                    new_data = {
                        "chosen": get_messages_from_debug_result(data["correct_solution"], wrong_solution["start_steps"]),
                        "rejected": get_messages_from_debug_result(wrong_solution["debug_result"], wrong_solution["start_steps"])
                    }
                    new_datas.append(new_data)
        print(f"{len(new_datas)}\n")
    new_datas = new_datas + get_chosen_rejected_alignment_lce_multi_file_diff(in_files_and_num)
    print(f"{len(new_datas)}\n")
    seed(3407)
    shuffle(new_datas)
    split_idx = int(len(new_datas) * 0.01)
    save_jsonl(new_datas[:split_idx], out_test_file)
    save_jsonl(new_datas[split_idx:], out_train_file)

def main_gsm8k():
    initial_file = "data/lce_solutions/mistral_lce_alignment_sample/gsm8k_train_lce_round53.jsonl"
    out_file = "data/lce_solutions/different_ranked_negative_divided/processed_results/gsm8k_train_lce_round53_step_controled_negative.jsonl"
    in_files = []
    for i in range(3):
        for j in range(100):
            in_file = f"data/lce_solutions/different_ranked_negative_divided/gsm8k/result_{i}_round{j}.jsonl"
            if os.path.isfile(in_file):
                in_files.append(in_file)
    prepare_correct_incorrect_solutions(initial_file, in_files, out_file)

def main_math():
    initial_file = "data/lce_solutions/mistral_lce_alignment_sample/math_train_lce_round7.jsonl"
    out_file = "data/lce_solutions/different_ranked_negative_divided/processed_results/math_train_lce_round7_step_controled_negative.jsonl"
    in_files = []
    for i in range(3):
        for j in range(100):
            in_file = f"data/lce_solutions/different_ranked_negative_divided/math/result_{i}_round{j}.jsonl"
            if os.path.isfile(in_file):
                in_files.append(in_file)
    prepare_correct_incorrect_solutions(initial_file, in_files, out_file)
    
def main_controled_steps():
    in_files = ["data/lce_solutions/different_ranked_negative_divided/processed_results/gsm8k_train_lce_round53_step_controled_negative.jsonl",
                "data/lce_solutions/different_ranked_negative_divided/processed_results/math_train_lce_round7_step_controled_negative.jsonl"]
    out_dir = f"data/controled_steps_math_gsm8k_lce_dpo"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/controled_steps_math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/controled_steps_math_gsm8k_test.jsonl"
    get_chosen_rejected_controled_steps(in_files, out_train_file, out_test_file)
    
def main_controled_steps_only_later_steps():
    in_files = ["data/lce_solutions/different_ranked_negative_divided/processed_results/gsm8k_train_lce_round53_step_controled_negative.jsonl",
                "data/lce_solutions/different_ranked_negative_divided/processed_results/math_train_lce_round7_step_controled_negative.jsonl"]
    out_dir = f"data/controled_steps_math_gsm8k_lce_dpo_only_later_steps"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/controled_steps_math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/controled_steps_math_gsm8k_test.jsonl"
    get_chosen_rejected_controled_steps_only_later_steps(in_files, out_train_file, out_test_file)

def main_controled_steps_naive():
    in_files = ["data/lce_solutions/different_ranked_negative_divided/processed_results/gsm8k_train_lce_round53_step_controled_negative.jsonl",
                "data/lce_solutions/different_ranked_negative_divided/processed_results/math_train_lce_round7_step_controled_negative.jsonl"]
    out_dir = f"data/controled_steps_math_gsm8k_lce_dpo_naive"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/controled_steps_math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/controled_steps_math_gsm8k_test.jsonl"
    get_chosen_rejected_controled_steps_naive(in_files, out_train_file, out_test_file)


def main_controled_steps_test_template():
    in_files = ["data/lce_solutions/different_ranked_negative_divided/processed_results/gsm8k_train_lce_round53_step_controled_negative.jsonl",
                "data/lce_solutions/different_ranked_negative_divided/processed_results/math_train_lce_round7_step_controled_negative.jsonl"]
    out_dir = f"data/controled_steps_math_gsm8k_lce_dpo_test_template"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/controled_steps_math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/controled_steps_math_gsm8k_test.jsonl"
    get_chosen_rejected_controled_steps_test_template(in_files, out_train_file, out_test_file)
    
def main_gsm8k_tmp1():
    initial_file = "data/lce_solutions/mistral_lce_alignment_sample/gsm8k_train_lce_round53.jsonl"
    out_file = "data/lce_solutions/different_ranked_negative_divided_tmp1/processed_results/gsm8k_train_lce_round53_step_controled_negative.jsonl"
    in_files = []
    for i in range(3):
        for j in range(100):
            in_file = f"data/lce_solutions/different_ranked_negative_divided_tmp1/gsm8k/result_{i}_round{j}.jsonl"
            if os.path.isfile(in_file):
                in_files.append(in_file)
    prepare_correct_incorrect_solutions(initial_file, in_files, out_file)

def main_math_tmp1():
    initial_file = "data/lce_solutions/mistral_lce_alignment_sample/math_train_lce_round7.jsonl"
    out_file = "data/lce_solutions/different_ranked_negative_divided_tmp1/processed_results/math_train_lce_round7_step_controled_negative.jsonl"
    in_files = []
    for i in range(3):
        for j in range(100):
            in_file = f"data/lce_solutions/different_ranked_negative_divided_tmp1/math/result_{i}_round{j}.jsonl"
            if os.path.isfile(in_file):
                in_files.append(in_file)
    prepare_correct_incorrect_solutions(initial_file, in_files, out_file)

def main_controled_steps_test_tmp1():
    in_files = ["data/lce_solutions/different_ranked_negative_divided_tmp1/processed_results/gsm8k_train_lce_round53_step_controled_negative.jsonl",
                "data/lce_solutions/different_ranked_negative_divided_tmp1/processed_results/math_train_lce_round7_step_controled_negative.jsonl"]
    out_dir = f"data/controled_steps_math_gsm8k_lce_dpo_tmp1"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/controled_steps_math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/controled_steps_math_gsm8k_test.jsonl"
    get_chosen_rejected_controled_steps(in_files, out_train_file, out_test_file)

def main_controled_steps_test_tmp13_tmp1():
    in_files = ["data/lce_solutions/different_ranked_negative_divided_tmp1/processed_results/gsm8k_train_lce_round53_step_controled_negative.jsonl",
                "data/lce_solutions/different_ranked_negative_divided_tmp1/processed_results/math_train_lce_round7_step_controled_negative.jsonl",
                "data/lce_solutions/different_ranked_negative_divided/processed_results/gsm8k_train_lce_round53_step_controled_negative.jsonl",
                "data/lce_solutions/different_ranked_negative_divided/processed_results/math_train_lce_round7_step_controled_negative.jsonl"]
    out_dir = f"data/controled_steps_math_gsm8k_lce_dpo_tmp13_tmp1"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/controled_steps_math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/controled_steps_math_gsm8k_test.jsonl"
    get_chosen_rejected_controled_steps(in_files, out_train_file, out_test_file)
    
def main_controled_steps_add_1x3_1x3():
    in_files = ["data/lce_solutions/different_ranked_negative_divided/processed_results/gsm8k_train_lce_round53_step_controled_negative.jsonl",
                "data/lce_solutions/different_ranked_negative_divided/processed_results/math_train_lce_round7_step_controled_negative.jsonl"]
    in_files_and_num = [("data/lce_solutions/mistral_lce_alignment_sample/gsm8k_train_lce_round53.jsonl", 1, 3),
                ("data/lce_solutions/mistral_lce_alignment_sample/math_train_lce_round7.jsonl", 1, 3)]
    out_dir = f"data/controled_steps_math_gsm8k_lce_dpo_add_1x3_1x3"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/controled_steps_math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/controled_steps_math_gsm8k_test.jsonl"
    get_chosen_rejected_controled_steps_add_1x3_1x3(in_files, in_files_and_num, out_train_file, out_test_file)
    
def main_gsm8k_ascend():
    initial_file = "data/lce_solutions/mistral_lce_alignment_sample/gsm8k_train_lce_round53.jsonl"
    out_file = "data/lce_solutions/ascending_temperature_negative/processed_results/gsm8k_train_lce_round53_step_controled_negative_1.jsonl"
    in_files = []
    for i in range(6):
        for j in range(100):
            in_file = f"data/lce_solutions/ascending_temperature_negative/gsm8k/result_{i}_round{j}.jsonl"
            if os.path.isfile(in_file):
                in_files.append(in_file)
    prepare_correct_incorrect_solutions(initial_file, in_files, out_file)

def main_math_ascend():
    initial_file = "data/lce_solutions/mistral_lce_alignment_sample/math_train_lce_round7.jsonl"
    out_file = "data/lce_solutions/ascending_temperature_negative/processed_results/math_train_lce_round7_step_controled_negative.jsonl"
    in_files = []
    for i in range(6):
        for j in range(100):
            in_file = f"data/lce_solutions/ascending_temperature_negative/math/result_{i}_round{j}.jsonl"
            if os.path.isfile(in_file):
                in_files.append(in_file)
    prepare_correct_incorrect_solutions(initial_file, in_files, out_file)
    
def main_controled_steps_ascend():
    in_files = ["data/lce_solutions/ascending_temperature_negative/processed_results/gsm8k_train_lce_round53_step_controled_negative.jsonl",
                "data/lce_solutions/ascending_temperature_negative/processed_results/math_train_lce_round7_step_controled_negative.jsonl"]
    neg_limit = 1000
    out_dir = f"data/controled_steps_math_gsm8k_lce_dpo_ascend_lim{neg_limit}"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/controled_steps_math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/controled_steps_math_gsm8k_test.jsonl"
    get_chosen_rejected_controled_steps_limited(in_files, out_train_file, out_test_file, neg_limit)

def main_controled_steps_ascend_1():
    neg_limit_gsm8k = 2
    neg_limit_math = 3
    in_files = [("data/lce_solutions/ascending_temperature_negative/processed_results/gsm8k_train_lce_round53_step_controled_negative.jsonl", neg_limit_gsm8k),
                ("data/lce_solutions/ascending_temperature_negative/processed_results/math_train_lce_round7_step_controled_negative.jsonl", neg_limit_math)]
    out_dir = f"data/controled_steps_math_gsm8k_lce_dpo_ascend_lim{neg_limit_gsm8k}_lim{neg_limit_math}"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/controled_steps_math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/controled_steps_math_gsm8k_test.jsonl"
    get_chosen_rejected_controled_steps_limited(in_files, out_train_file, out_test_file)

def main_controled_steps_ascend_add_dpo1x1():
    neg_limit_gsm8k = 2
    neg_limit_math = 3
    in_files_and_num = [("data/lce_solutions/mistral_lce_alignment_sample/gsm8k_train_lce_round53.jsonl", 1, 1),
                ("data/lce_solutions/mistral_lce_alignment_sample/math_train_lce_round7.jsonl", 1, 1)]
    in_files = [("data/lce_solutions/ascending_temperature_negative/processed_results/gsm8k_train_lce_round53_step_controled_negative.jsonl", neg_limit_gsm8k),
                ("data/lce_solutions/ascending_temperature_negative/processed_results/math_train_lce_round7_step_controled_negative.jsonl", neg_limit_math)]
    out_dir = f"data/controled_steps_math_gsm8k_lce_dpo_ascend_lim{neg_limit_gsm8k}_lim{neg_limit_math}_add_dpo1x1"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/controled_steps_math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/controled_steps_math_gsm8k_test.jsonl"
    get_chosen_rejected_controled_steps_limited_add_dpo1x1(in_files, in_files_and_num, out_train_file, out_test_file)

    
def main_controled_steps_ascend_add_dpo1x1_1():
    neg_limit_gsm8k = 2
    neg_limit_math = 3
    in_files_and_num = [("data/lce_solutions/mistral_lce_alignment_sample/gsm8k_train_lce_round53.jsonl", 1, 1),
                ("data/lce_solutions/mistral_lce_alignment_sample/math_train_lce_round7.jsonl", 1, 1)]
    in_files = [("data/lce_solutions/ascending_temperature_negative/processed_results/gsm8k_train_lce_round53_step_controled_negative_1.jsonl", neg_limit_gsm8k),
                ("data/lce_solutions/ascending_temperature_negative/processed_results/math_train_lce_round7_step_controled_negative.jsonl", neg_limit_math)]
    out_dir = f"data/controled_steps_math_gsm8k_lce_dpo_ascend_lim{neg_limit_gsm8k}_lim{neg_limit_math}_add_dpo1x1_1"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/controled_steps_math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/controled_steps_math_gsm8k_test.jsonl"
    get_chosen_rejected_controled_steps_limited_add_dpo1x1(in_files, in_files_and_num, out_train_file, out_test_file)

def main_controled_steps_ascend_add_dpo1x1_2():
    neg_limit_gsm8k = 2
    neg_limit_math = 3
    in_files_and_num = [("data/lce_solutions/mistral_lce_alignment_sample_1/processed_results/gsm8k_results_7473.jsonl", 1, 1),
                ("data/lce_solutions/mistral_lce_alignment_sample_1/processed_results/math_results_7500.jsonl", 1, 1)]
    in_files = [("data/lce_solutions/ascending_temperature_negative/processed_results/gsm8k_train_lce_round53_step_controled_negative.jsonl", neg_limit_gsm8k),
                ("data/lce_solutions/ascending_temperature_negative/processed_results/math_train_lce_round7_step_controled_negative.jsonl", neg_limit_math)]
    out_dir = f"data/controled_steps_math_gsm8k_lce_dpo_ascend_lim{neg_limit_gsm8k}_lim{neg_limit_math}_add_dpo1x1_2"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/controled_steps_math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/controled_steps_math_gsm8k_test.jsonl"
    get_chosen_rejected_controled_steps_limited_add_dpo1x1(in_files, in_files_and_num, out_train_file, out_test_file)

def main_controled_steps_ascend_add_dpo1x1_1x2():
    neg_limit_gsm8k = 2
    neg_limit_math = 3
    in_files_and_num = [("data/lce_solutions/mistral_lce_alignment_sample_1/processed_results/gsm8k_results_7473.jsonl", 1, 1),
                ("data/lce_solutions/mistral_lce_alignment_sample_1/processed_results/math_results_7500.jsonl", 1, 2)]
    in_files = [("data/lce_solutions/ascending_temperature_negative/processed_results/gsm8k_train_lce_round53_step_controled_negative.jsonl", neg_limit_gsm8k),
                ("data/lce_solutions/ascending_temperature_negative/processed_results/math_train_lce_round7_step_controled_negative.jsonl", neg_limit_math)]
    out_dir = f"data/controled_steps_math_gsm8k_lce_dpo_ascend_lim{neg_limit_gsm8k}_lim{neg_limit_math}_add_dpo1x1_1x2"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/controled_steps_math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/controled_steps_math_gsm8k_test.jsonl"
    get_chosen_rejected_controled_steps_limited_add_dpo1x1(in_files, in_files_and_num, out_train_file, out_test_file)
    
def main_gsm8k_no_ch_internlm():
    initial_file = "data/lce_solutions/internlm_no_ch_negative_positive_gen/naive_dpo/processed_results/gsm8k_results_7473.jsonl"
    out_file = "data/lce_solutions/internlm_no_ch_negative_positive_gen/sc_dpo/processed_results/gsm8k_step_controled_negative.jsonl"
    in_files = []
    for i in range(18):
        for j in range(300):
            in_file = f"data/lce_solutions/internlm_no_ch_negative_positive_gen/sc_dpo/gsm8k/result_{i}_round{j}.jsonl"
            if os.path.isfile(in_file):
                in_files.append(in_file)
    prepare_correct_incorrect_solutions(initial_file, in_files, out_file)
    
def main_math_no_ch_internlm():
    initial_file = "data/lce_solutions/internlm_no_ch_negative_positive_gen/naive_dpo/processed_results/math_results_7500.jsonl"
    out_file = "data/lce_solutions/internlm_no_ch_negative_positive_gen/sc_dpo/processed_results/math_step_controled_negative.jsonl"
    in_files = []
    for i in range(18):
        for j in range(300):
            in_file = f"data/lce_solutions/internlm_no_ch_negative_positive_gen/sc_dpo/math/result_{i}_round{j}.jsonl"
            if os.path.isfile(in_file):
                in_files.append(in_file)
    prepare_correct_incorrect_solutions(initial_file, in_files, out_file)
    
def main_controled_steps_ascend_add_dpo1x1():
    neg_limit_gsm8k = 2
    neg_limit_math = 3
    in_files_and_num = [("data/lce_solutions/internlm_no_ch_negative_positive_gen/naive_dpo/processed_results/gsm8k_results_7473.jsonl", 1, 1),
                ("data/lce_solutions/internlm_no_ch_negative_positive_gen/naive_dpo/processed_results/math_results_7500.jsonl", 1, 1)]
    in_files = [("data/lce_solutions/internlm_no_ch_negative_positive_gen/sc_dpo/processed_results/gsm8k_step_controled_negative.jsonl", neg_limit_gsm8k),
                ("data/lce_solutions/internlm_no_ch_negative_positive_gen/sc_dpo/processed_results/math_step_controled_negative.jsonl", neg_limit_math)]
    out_dir = f"data/internlm_controled_steps_math_gsm8k_lce_dpo_ascend_lim{neg_limit_gsm8k}_lim{neg_limit_math}_add_dpo1x1"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/controled_steps_math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/controled_steps_math_gsm8k_test.jsonl"
    get_chosen_rejected_controled_steps_limited_add_dpo1x1(in_files, in_files_and_num, out_train_file, out_test_file)
    
def main_gsm8k_internlm():
    initial_file = "data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/gsm8k_results_7473.jsonl"
    out_file = "data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/gsm8k_step_controled_negative.jsonl"
    in_files = []
    for i in range(18):
        for j in range(300):
            in_file = f"data/lce_solutions/internlm_negative_positive_gen/sc_dpo/gsm8k/result_{i}_round{j}.jsonl"
            if os.path.isfile(in_file):
                in_files.append(in_file)
    prepare_correct_incorrect_solutions(initial_file, in_files, out_file)
    
def main_math_internlm():
    initial_file = "data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/math_results_7500.jsonl"
    out_file = "data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/math_step_controled_negative.jsonl"
    in_files = []
    for i in range(18):
        for j in range(300):
            in_file = f"data/lce_solutions/internlm_negative_positive_gen/sc_dpo/math/result_{i}_round{j}.jsonl"
            if os.path.isfile(in_file):
                in_files.append(in_file)
    prepare_correct_incorrect_solutions(initial_file, in_files, out_file)
    
def main_ape_internlm():
    initial_file = "data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/ape_results_200488.jsonl"
    out_file = "data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/ape_step_controled_negative.jsonl"
    in_files = []
    for i in range(18):
        for j in range(300):
            in_file = f"data/lce_solutions/internlm_negative_positive_gen/sc_dpo/ape/result_{i}_round{j}.jsonl"
            if os.path.isfile(in_file):
                in_files.append(in_file)
    prepare_correct_incorrect_solutions(initial_file, in_files, out_file)
    
def main_gsm8k_math_ape_lim2_3_controled_steps_ascend_add_dpo1x1_1x4_1x1():
    neg_limit_gsm8k = 2
    neg_limit_math = 3
    neg_limit_ape = 2
    in_files_and_num = [("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/gsm8k_results_7473.jsonl", 1, 1),
                ("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/math_results_7500.jsonl", 1, 4),
                ("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/ape_results_200488.jsonl", 1, 1)]
    in_files = [("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/gsm8k_step_controled_negative.jsonl", neg_limit_gsm8k),
                ("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/math_step_controled_negative.jsonl", neg_limit_math),
                ("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/ape_step_controled_negative.jsonl", neg_limit_ape)]
    out_dir = f"data/internlm_controled_steps_math_gsm8k_quarterape_lce_dpo_ascend_lim{neg_limit_gsm8k}_lim{neg_limit_math}_lim{neg_limit_ape}"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/controled_steps_math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/controled_steps_math_gsm8k_test.jsonl"
    get_chosen_rejected_controled_steps_limited_add_dpo1x1(in_files, in_files_and_num, out_train_file, out_test_file, weights=(1, 1, 0.25))
    
def main_gsm8k_math_ape_lim2_3_2_controled_steps_ascend_add_dpo1x1_1x4_1x1():
    neg_limit_gsm8k = 2
    neg_limit_math = 3
    neg_limit_ape = 2
    in_files_and_num = [("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/gsm8k_results_7473.jsonl", 1, 1),
                ("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/math_results_7500.jsonl", 1, 4),
                ("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/ape_results_200488.jsonl", 1, 1)]
    in_files = [("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/gsm8k_step_controled_negative.jsonl", neg_limit_gsm8k),
                ("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/math_step_controled_negative.jsonl", neg_limit_math),
                ("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/ape_step_controled_negative.jsonl", neg_limit_ape)]
    out_dir = f"data/internlm_controled_steps_math_gsm8k_quarterape_lce_dpo_ascend_lim{neg_limit_gsm8k}_lim{neg_limit_math}_lim{neg_limit_ape}"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/controled_steps_math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/controled_steps_math_gsm8k_test.jsonl"
    get_chosen_rejected_controled_steps_limited_add_dpo1x1(in_files, in_files_and_num, out_train_file, out_test_file, weights=(1, 1, 0.25))

def main_gsm8k_math_ape_lim2_3_controled_steps_ascend_add_dpo1x1_1x4_1x1():
    neg_limit_gsm8k = 2
    neg_limit_math = 3
    in_files_and_num = [("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/gsm8k_results_7473.jsonl", 1, 1),
                ("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/math_results_7500.jsonl", 1, 4),
                ("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/ape_results_200488.jsonl", 1, 1)]
    in_files = [("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/gsm8k_step_controled_negative.jsonl", neg_limit_gsm8k),
                ("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/math_step_controled_negative.jsonl", neg_limit_math)]
    out_dir = f"data/internlm_controled_steps_math_gsm8k_quarterape_lce_dpo_ascend_lim{neg_limit_gsm8k}_lim{neg_limit_math}"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/controled_steps_math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/controled_steps_math_gsm8k_test.jsonl"
    get_chosen_rejected_controled_steps_limited_add_dpo1x1(in_files, in_files_and_num, out_train_file, out_test_file, weights=(1, 1, 0.25))
    
def main_gsm8k_math_ape_lim2_3_1half_controled_steps_ascend_add_dpo1x1_1x4_1x1():
    neg_limit_gsm8k = 2
    neg_limit_math = 3
    neg_limit_ape = 1
    in_files_and_num = [("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/gsm8k_results_7473.jsonl", 1, 1),
                ("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/math_results_7500.jsonl", 1, 4),
                ("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/ape_results_200488.jsonl", 1, 1)]
    in_files = [("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/gsm8k_step_controled_negative.jsonl", neg_limit_gsm8k),
                ("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/math_step_controled_negative.jsonl", neg_limit_math),
                ("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/ape_step_controled_negative.jsonl", neg_limit_ape)]
    out_dir = f"data/internlm_controled_steps_math_gsm8k_quarterape_lce_dpo_ascend_lim{neg_limit_gsm8k}_lim{neg_limit_math}_lim{neg_limit_ape}half"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/controled_steps_math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/controled_steps_math_gsm8k_test.jsonl"
    get_chosen_rejected_controled_steps_limited_add_dpo1x1(in_files, in_files_and_num, out_train_file, out_test_file, weights=(1, 1, 0.25), weights_sc=(1, 1, 0.5))
    
def main_gsm8k_math_ape_lim2_3_1threequarter_controled_steps_ascend_add_dpo1x1_1x4_1x1():
    neg_limit_gsm8k = 2
    neg_limit_math = 3
    neg_limit_ape = 1
    in_files_and_num = [("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/gsm8k_results_7473.jsonl", 1, 1),
                ("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/math_results_7500.jsonl", 1, 4),
                ("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/ape_results_200488.jsonl", 1, 1)]
    in_files = [("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/gsm8k_step_controled_negative.jsonl", neg_limit_gsm8k),
                ("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/math_step_controled_negative.jsonl", neg_limit_math),
                ("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/ape_step_controled_negative.jsonl", neg_limit_ape)]
    out_dir = f"data/internlm_controled_steps_math_gsm8k_quarterape_lce_dpo_ascend_lim{neg_limit_gsm8k}_lim{neg_limit_math}_lim{neg_limit_ape}threequarter"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/controled_steps_math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/controled_steps_math_gsm8k_test.jsonl"
    get_chosen_rejected_controled_steps_limited_add_dpo1x1(in_files, in_files_and_num, out_train_file, out_test_file, weights=(1, 1, 0.25), weights_sc=(1, 1, 0.75))

def main_gsm8k_math_ape_lim2_3_1quarter_controled_steps_ascend_add_dpo1x1_1x4_1x1():
    neg_limit_gsm8k = 2
    neg_limit_math = 3
    neg_limit_ape = 1
    in_files_and_num = [("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/gsm8k_results_7473.jsonl", 1, 1),
                ("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/math_results_7500.jsonl", 1, 4),
                ("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/ape_results_200488.jsonl", 1, 1)]
    in_files = [("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/gsm8k_step_controled_negative.jsonl", neg_limit_gsm8k),
                ("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/math_step_controled_negative.jsonl", neg_limit_math),
                ("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/ape_step_controled_negative.jsonl", neg_limit_ape)]
    out_dir = f"data/internlm_controled_steps_math_gsm8k_quarterape_lce_dpo_ascend_lim{neg_limit_gsm8k}_lim{neg_limit_math}_lim{neg_limit_ape}quarter"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/controled_steps_math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/controled_steps_math_gsm8k_test.jsonl"
    get_chosen_rejected_controled_steps_limited_add_dpo1x1(in_files, in_files_and_num, out_train_file, out_test_file, weights=(1, 1, 0.25), weights_sc=(1, 1, 0.25))
    
def main_math_internlm_1():
    initial_file = "data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/math_results_7500.jsonl"
    out_file = "data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/math_step_controled_negative_1.jsonl"
    in_files = []
    for i in range(18):
        for j in range(300):
            in_file = f"data/lce_solutions/internlm_negative_positive_gen/sc_dpo/math/result_{i}_round{j}.jsonl"
            if os.path.isfile(in_file):
                in_files.append(in_file)
    prepare_correct_incorrect_solutions(initial_file, in_files, out_file)
    
def main_gsm8k_math_ape_lim2_3_1half_controled_steps_ascend_add_dpo1x1_1x4_1x1_1():
    neg_limit_gsm8k = 2
    neg_limit_math = 3
    neg_limit_ape = 1
    in_files_and_num = [("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/gsm8k_results_7473.jsonl", 1, 1),
                ("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/math_results_7500.jsonl", 1, 4),
                ("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/ape_results_200488.jsonl", 1, 1)]
    in_files = [("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/gsm8k_step_controled_negative.jsonl", neg_limit_gsm8k),
                ("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/math_step_controled_negative_1.jsonl", neg_limit_math),
                ("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/ape_step_controled_negative.jsonl", neg_limit_ape)]
    out_dir = f"data/internlm_controled_steps_math_gsm8k_quarterape_lce_dpo_ascend_lim{neg_limit_gsm8k}_lim{neg_limit_math}_lim{neg_limit_ape}half_1"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/controled_steps_math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/controled_steps_math_gsm8k_test.jsonl"
    get_chosen_rejected_controled_steps_limited_add_dpo1x1(in_files, in_files_and_num, out_train_file, out_test_file, weights=(1, 1, 0.25), weights_sc=(1, 1, 0.5))
    
def main_gsm8k_math_ape_lim2_3_1quarter_controled_steps_ascend_add_dpo1x1_1x4_1x1_1():
    neg_limit_gsm8k = 2
    neg_limit_math = 3
    neg_limit_ape = 1
    in_files_and_num = [("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/gsm8k_results_7473.jsonl", 1, 1),
                ("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/math_results_7500.jsonl", 1, 4),
                ("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/ape_results_200488.jsonl", 1, 1)]
    in_files = [("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/gsm8k_step_controled_negative.jsonl", neg_limit_gsm8k),
                ("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/math_step_controled_negative_1.jsonl", neg_limit_math),
                ("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/ape_step_controled_negative.jsonl", neg_limit_ape)]
    out_dir = f"data/internlm_controled_steps_math_gsm8k_quarterape_lce_dpo_ascend_lim{neg_limit_gsm8k}_lim{neg_limit_math}_lim{neg_limit_ape}quarter_1"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/controled_steps_math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/controled_steps_math_gsm8k_test.jsonl"
    get_chosen_rejected_controled_steps_limited_add_dpo1x1(in_files, in_files_and_num, out_train_file, out_test_file, weights=(1, 1, 0.25), weights_sc=(1, 1, 0.25))
    
    
def main_gsm8k_math_ape_lim2_3_1half_3_controled_steps_ascend_add_dpo1x1_1x4_1x1_1():
    neg_limit_gsm8k = 2
    neg_limit_math = 3
    neg_limit_ape = 1
    in_files_and_num = [("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/gsm8k_results_7473.jsonl", 1, 1),
                ("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/math_results_7500.jsonl", 1, 4),
                ("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/ape_results_200488.jsonl", 1, 1)]
    in_files = [("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/gsm8k_step_controled_negative.jsonl", neg_limit_gsm8k),
                ("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/math_step_controled_negative_1.jsonl", neg_limit_math),
                ("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/ape_step_controled_negative.jsonl", neg_limit_ape),
                ("data/lce_solutions/internlm_no_ch_negative_positive_gen/sc_dpo/processed_results/math_step_controled_negative.jsonl", neg_limit_math)]
    out_dir = f"data/internlm_controled_steps_math_gsm8k_quarterape_lce_dpo_ascend_lim{neg_limit_gsm8k}_lim{neg_limit_math}_lim{neg_limit_ape}half_mathlim{neg_limit_math}_1"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/controled_steps_math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/controled_steps_math_gsm8k_test.jsonl"
    get_chosen_rejected_controled_steps_limited_add_dpo1x1(in_files, in_files_and_num, out_train_file, out_test_file, weights=(1, 1, 0.25), weights_sc=(1, 1, 0.5, 1))

def main_gsm8k_math_ape_lim2_3_1quarter_controled_steps_ascend_add_dpo1x1_1x1_1x1_1():
    neg_limit_gsm8k = 2
    neg_limit_math = 3
    neg_limit_ape = 1
    in_files_and_num = [("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/gsm8k_results_7473.jsonl", 1, 1),
                ("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/math_results_7500.jsonl", 1, 1),
                ("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/ape_results_200488.jsonl", 1, 1)]
    in_files = [("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/gsm8k_step_controled_negative.jsonl", neg_limit_gsm8k),
                ("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/math_step_controled_negative_1.jsonl", neg_limit_math),
                ("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/ape_step_controled_negative.jsonl", neg_limit_ape)]
    out_dir = f"data/internlm_controled_steps_math1x1_gsm8k_quarterape_lce_dpo_ascend_lim{neg_limit_gsm8k}_lim{neg_limit_math}_lim{neg_limit_ape}quarter_1"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/controled_steps_math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/controled_steps_math_gsm8k_test.jsonl"
    get_chosen_rejected_controled_steps_limited_add_dpo1x1(in_files, in_files_and_num, out_train_file, out_test_file, weights=(1, 1, 0.25), weights_sc=(1, 1, 0.25))
    
def main_gsm8k_math_ape_lim2_3_1half_math3_gsm8k2_controled_steps_ascend_add_dpo1x1_1x4_1x1_1():
    neg_limit_gsm8k = 2
    neg_limit_math = 3
    neg_limit_ape = 1
    in_files_and_num = [("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/gsm8k_results_7473.jsonl", 1, 1),
                ("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/math_results_7500.jsonl", 1, 4),
                ("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/ape_results_200488.jsonl", 1, 1)]
    in_files = [("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/gsm8k_step_controled_negative.jsonl", neg_limit_gsm8k),
                ("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/math_step_controled_negative_1.jsonl", neg_limit_math),
                ("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/ape_step_controled_negative.jsonl", neg_limit_ape),
                ("data/lce_solutions/internlm_no_ch_negative_positive_gen/sc_dpo/processed_results/math_step_controled_negative.jsonl", neg_limit_math),
                ("data/lce_solutions/internlm_no_ch_negative_positive_gen/sc_dpo/processed_results/gsm8k_step_controled_negative.jsonl", neg_limit_gsm8k)]
    out_dir = f"data/internlm_controled_steps_math_gsm8k_quarterape_lce_dpo_ascend_lim{neg_limit_gsm8k}_lim{neg_limit_math}_lim{neg_limit_ape}half_mathlim{neg_limit_math}_gsm8klim{neg_limit_gsm8k}_1"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/controled_steps_math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/controled_steps_math_gsm8k_test.jsonl"
    get_chosen_rejected_controled_steps_limited_add_dpo1x1(in_files, in_files_and_num, out_train_file, out_test_file, weights=(1, 1, 0.25), weights_sc=(1, 1, 0.5, 1, 1))
    
def main_gsm8k_math_ape_lim2_3_1half_controled_steps_ascend_add_dpo1x1_1x1_1x1_1():
    neg_limit_gsm8k = 2
    neg_limit_math = 3
    neg_limit_ape = 1
    in_files_and_num = [("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/gsm8k_results_7473.jsonl", 1, 1),
                ("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/math_results_7500.jsonl", 1, 1),
                ("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/ape_results_200488.jsonl", 1, 1)]
    in_files = [("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/gsm8k_step_controled_negative.jsonl", neg_limit_gsm8k),
                ("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/math_step_controled_negative_1.jsonl", neg_limit_math),
                ("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/ape_step_controled_negative.jsonl", neg_limit_ape)]
    out_dir = f"data/internlm_controled_steps_math1x1_gsm8k_quarterape_lce_dpo_ascend_lim{neg_limit_gsm8k}_lim{neg_limit_math}_lim{neg_limit_ape}half_1"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/controled_steps_math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/controled_steps_math_gsm8k_test.jsonl"
    get_chosen_rejected_controled_steps_limited_add_dpo1x1(in_files, in_files_and_num, out_train_file, out_test_file, weights=(1, 1, 0.25), weights_sc=(1, 1, 0.5))

def main_gsm8k_math_ape_lim2_3_1half_math3_gsm8k2half_controled_steps_ascend_add_dpo1x1_1x4_1x1_1():
    neg_limit_gsm8k = 2
    neg_limit_math = 3
    neg_limit_ape = 1
    in_files_and_num = [("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/gsm8k_results_7473.jsonl", 1, 1),
                ("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/math_results_7500.jsonl", 1, 4),
                ("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/ape_results_200488.jsonl", 1, 1)]
    in_files = [("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/gsm8k_step_controled_negative.jsonl", neg_limit_gsm8k),
                ("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/math_step_controled_negative_1.jsonl", neg_limit_math),
                ("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/ape_step_controled_negative.jsonl", neg_limit_ape),
                ("data/lce_solutions/internlm_no_ch_negative_positive_gen/sc_dpo/processed_results/math_step_controled_negative.jsonl", neg_limit_math),
                ("data/lce_solutions/internlm_no_ch_negative_positive_gen/sc_dpo/processed_results/gsm8k_step_controled_negative.jsonl", neg_limit_gsm8k)]
    out_dir = f"data/internlm_controled_steps_math_gsm8k_quarterape_lce_dpo_ascend_lim{neg_limit_gsm8k}_lim{neg_limit_math}_lim{neg_limit_ape}half_mathlim{neg_limit_math}_gsm8klim{neg_limit_gsm8k}half_1"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/controled_steps_math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/controled_steps_math_gsm8k_test.jsonl"
    get_chosen_rejected_controled_steps_limited_add_dpo1x1(in_files, in_files_and_num, out_train_file, out_test_file, weights=(1, 1, 0.25), weights_sc=(1, 1, 0.5, 1, 0.5))

def main_gsm8k_math_ape_lim2_3_2_math3_gsm8k2_controled_steps_ascend_add_dpo1x1_1x4_1x1_1():
    neg_limit_gsm8k = 2
    neg_limit_math = 3
    neg_limit_ape = 2
    in_files_and_num = [("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/gsm8k_results_7473.jsonl", 1, 1),
                ("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/math_results_7500.jsonl", 1, 4),
                ("data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/ape_results_200488.jsonl", 1, 1)]
    in_files = [("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/gsm8k_step_controled_negative.jsonl", neg_limit_gsm8k),
                ("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/math_step_controled_negative_1.jsonl", neg_limit_math),
                ("data/lce_solutions/internlm_negative_positive_gen/sc_dpo/processed_results/ape_step_controled_negative.jsonl", neg_limit_ape),
                ("data/lce_solutions/internlm_no_ch_negative_positive_gen/sc_dpo/processed_results/math_step_controled_negative.jsonl", neg_limit_math),
                ("data/lce_solutions/internlm_no_ch_negative_positive_gen/sc_dpo/processed_results/gsm8k_step_controled_negative.jsonl", neg_limit_gsm8k)]
    out_dir = f"data/internlm_controled_steps_math_gsm8k_quarterape_lce_dpo_ascend_lim{neg_limit_gsm8k}_lim{neg_limit_math}_lim{neg_limit_ape}_mathlim{neg_limit_math}_gsm8klim{neg_limit_gsm8k}_1"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/controled_steps_math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/controled_steps_math_gsm8k_test.jsonl"
    get_chosen_rejected_controled_steps_limited_add_dpo1x1(in_files, in_files_and_num, out_train_file, out_test_file, weights=(1, 1, 0.25), weights_sc=(1, 1, 1, 1, 1))

def main_controled_steps_ascend_add_dpo1x1_gsm8k_only():
    neg_limit_gsm8k = 2
    neg_limit_math = 3
    in_files_and_num = [("data/lce_solutions/mistral_lce_alignment_sample/gsm8k_train_lce_round53.jsonl", 1, 1)]
    in_files = [("data/lce_solutions/ascending_temperature_negative/processed_results/gsm8k_train_lce_round53_step_controled_negative.jsonl", neg_limit_gsm8k)]
    out_dir = f"data/controled_steps_gsm8k_only_lce_dpo_ascend_lim{neg_limit_gsm8k}_add_dpo1x1"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/controled_steps_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/controled_steps_gsm8k_test.jsonl"
    get_chosen_rejected_controled_steps_limited_add_dpo1x1(in_files, in_files_and_num, out_train_file, out_test_file)
    
def main_controled_steps_ascend_add_dpo1x1_math_only():
    neg_limit_gsm8k = 2
    neg_limit_math = 3
    in_files_and_num = [("data/lce_solutions/mistral_lce_alignment_sample/math_train_lce_round7.jsonl", 1, 1)]
    in_files = [("data/lce_solutions/ascending_temperature_negative/processed_results/math_train_lce_round7_step_controled_negative.jsonl", neg_limit_math)]
    out_dir = f"data/controled_steps_math_only_lce_dpo_ascend_lim{neg_limit_math}_add_dpo1x1"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/controled_steps_math_train.jsonl"
    out_test_file = f"{out_dir}/data/test/controled_steps_math_test.jsonl"
    get_chosen_rejected_controled_steps_limited_add_dpo1x1(in_files, in_files_and_num, out_train_file, out_test_file)

if __name__ == "__main__":
    main_controled_steps_ascend_add_dpo1x1_gsm8k_only()
    main_controled_steps_ascend_add_dpo1x1_math_only()