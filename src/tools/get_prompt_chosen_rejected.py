import json
import os
import sys
from tqdm import tqdm
from random import shuffle, seed
from argparse import ArgumentParser

def save_jsonl(datas, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        for data in datas:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

def load_jsonl(in_file):
    with open(in_file, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]
    return datas

def get_target_from_debug_result(debug_result):
    target = ""
    for block in debug_result[2:]:
        if block["role"] == "text":
            target += f"<|text|>{block['content']}<|endofblock|>"
        elif block["role"] == "code":
            target += f"<|code|>{block['content']}<|endofblock|>"
        elif block["role"] == "execution":
            target += f"<|execution|>{block['content']}<|endofblock|>"
    return f"<|assistant|>{target}<|endofmessage|>"

def get_prompt_chosen_rejected_single(data):
    if len(data["correct_solutions"]) > 0 and len(data["wrong_solutions"]) > 0:
        prompt = f"<|system|><|text|><|endofblock|><|endofmessage|><|user|><|text|>{data['question']}<|endofblock|><|endofmessage|>"
        chosen = get_target_from_debug_result(data["correct_solutions"][0])
        rejected = get_target_from_debug_result(data["wrong_solutions"][0])
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    else:
        return None

def get_prompt_chosen_rejected(in_files, out_file):
    seed(3407)
    new_datas = []
    for in_file in tqdm(in_files):
        datas = load_jsonl(in_file)
        for data in tqdm(datas):
            new_data = get_prompt_chosen_rejected_single(data)
            if new_data is not None:
                new_datas.append(get_prompt_chosen_rejected_single(data))
        print(len(new_datas))
    print(f"num_chosen_rejected_pairs: {len(new_datas)}")
    shuffle(new_datas)
    save_jsonl(new_datas, out_file)
    
def get_messages_from_debug_result(debug_result):
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

def get_chosen_rejected_alignment_lce(in_files, out_train_file, out_test_file):
    new_datas = []
    for in_file in in_files:
        datas = load_jsonl(in_file)
        for data in tqdm(datas):
            if len(data["correct_solutions"]) > 0 and len(data["wrong_solutions"]) > 0:
                new_data = {
                    "chosen": get_messages_from_debug_result(data["correct_solutions"][0]),
                    "rejected": get_messages_from_debug_result(data["wrong_solutions"][0])
                }
                new_datas.append(new_data)
        print(f"{len(new_datas)}\n")
    seed(3407)
    shuffle(new_datas)
    split_idx = int(len(new_datas) * 0.01)
    save_jsonl(new_datas[:split_idx], out_test_file)
    save_jsonl(new_datas[split_idx:], out_train_file)
    
def get_chosen_rejected_alignment_lce_multi(in_files, out_train_file, out_test_file, num_correct, num_wrong):
    new_datas = []
    for in_file in in_files:
        datas = load_jsonl(in_file)
        for data in tqdm(datas):
            for correct_solution in data["correct_solutions"][:num_correct]:
                for wrong_solution in data["wrong_solutions"][:num_wrong]:
                    new_data = {
                        "chosen": get_messages_from_debug_result(correct_solution),
                        "rejected": get_messages_from_debug_result(wrong_solution)
                    }
                    new_datas.append(new_data)
        print(f"{len(new_datas)}\n")
    seed(3407)
    shuffle(new_datas)
    split_idx = int(len(new_datas) * 0.01)
    save_jsonl(new_datas[:split_idx], out_test_file)
    save_jsonl(new_datas[split_idx:], out_train_file)
    
def get_chosen_rejected_alignment_lce_multi_file_diff(in_files_and_num, out_train_file, out_test_file, weight=(1, 1, 1)):
    new_datas = []
    pre_len = 0
    idx = 0
    for in_file, num_correct, num_wrong in in_files_and_num:
        datas = load_jsonl(in_file)
        length = len(datas)
        datas = datas[:int(length * weight[idx])]
        idx += 1
        for data in tqdm(datas):
            for correct_solution in data["correct_solutions"][:num_correct]:
                cnt = num_wrong
                for wrong_solution in data["wrong_solutions"]:
                    if len(wrong_solution) > 3:
                        new_data = {
                            "chosen": get_messages_from_debug_result(correct_solution),
                            "rejected": get_messages_from_debug_result(wrong_solution)
                        }
                        new_datas.append(new_data)
                        cnt -= 1
                    else:
                        continue
                    if cnt == 0:
                        break
        print(f"{in_file}: {len(new_datas) - pre_len}")
        print(len(new_datas))
        pre_len = len(new_datas)
    seed(3407)
    shuffle(new_datas)
    split_idx = int(len(new_datas) * 0.01)
    save_jsonl(new_datas[:split_idx], out_test_file)
    save_jsonl(new_datas[split_idx:], out_train_file)
    
def get_chosen_rejected_alignment_lce_multi_file_diff_add_neg_from_other(in_files_and_num, out_train_file, out_test_file, weight=(1, 1, 1)):
    new_datas = []
    pre_len = 0
    idx = 0
    for in_files, num_correct, num_wrong in in_files_and_num:
        datas = load_jsonl(in_files[0])
        if len(in_files) > 1:
            for in_file in in_files[1:]:
                datas_extra_neg = load_jsonl(in_file)
                for i in range(len(datas)):
                    datas[i]["wrong_solutions"].extend(datas_extra_neg[i]["wrong_solutions"])
        length = len(datas)
        datas = datas[:int(length * weight[idx])]
        idx += 1
        for data in tqdm(datas):
            for correct_solution in data["correct_solutions"][:num_correct]:
                cnt = num_wrong
                for wrong_solution in data["wrong_solutions"]:
                    if len(wrong_solution) > 3:
                        new_data = {
                            "chosen": get_messages_from_debug_result(correct_solution),
                            "rejected": get_messages_from_debug_result(wrong_solution)
                        }
                        new_datas.append(new_data)
                        cnt -= 1
                    else:
                        continue
                    if cnt == 0:
                        break
        print(f"{in_file}: {len(new_datas) - pre_len}")
        print(len(new_datas))
        pre_len = len(new_datas)
    seed(3407)
    shuffle(new_datas)
    split_idx = int(len(new_datas) * 0.01)
    save_jsonl(new_datas[:split_idx], out_test_file)
    save_jsonl(new_datas[split_idx:], out_train_file)
    
def get_chosen_rejected_alignment_lce_multi_file_diff_prefer_long(in_files_and_num, out_train_file, out_test_file):
    new_datas = []
    for in_file, num_correct, num_wrong in in_files_and_num:
        datas = load_jsonl(in_file)
        for data in tqdm(datas):
            correct_solutions = sorted(data["correct_solutions"], key=lambda d: len(d), reverse=True)
            for correct_solution in correct_solutions[:num_correct]:
                for wrong_solution in data["wrong_solutions"][:num_wrong]:
                    new_data = {
                        "chosen": get_messages_from_debug_result(correct_solution),
                        "rejected": get_messages_from_debug_result(wrong_solution)
                    }
                    new_datas.append(new_data)
        print(f"{len(new_datas)}\n")
    seed(3407)
    shuffle(new_datas)
    split_idx = int(len(new_datas) * 0.01)
    save_jsonl(new_datas[:split_idx], out_test_file)
    save_jsonl(new_datas[split_idx:], out_train_file)
    
def count_valid_num(in_file, correct_num, wrong_num):
    datas = load_jsonl(in_file)
    valid_num = 0
    for data in tqdm(datas):
        if len(data["correct_solutions"]) >= correct_num and len(data["wrong_solutions"]) >= wrong_num:
            valid_num += 1
            
    print(f"valid_num: {valid_num}")

def main():
    in_files = ["/mnt/cache/luzimu/rlhf_math/data/lce_solutions/negative_positive_sample/gsm8k_train_lce_round2.jsonl",
    "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/negative_positive_sample/math_train_lce_round2.jsonl"]
    out_file = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/chosen_rejected_data/gsm8k_math_chosen_rejected_iter1.jsonl"
    get_prompt_chosen_rejected(in_files, out_file)
    
def main1():
    in_files = ["/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mistral_lce_alignment_sample_it2/gsm8k_train_lce_round80.jsonl",
                "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mistral_lce_alignment_sample_it2/math_train_lce_round8.jsonl"]
    out_train_file = "/mnt/cache/luzimu/rlhf_math/data/math_gsm8k_lce_dpo_it2/data/train/math_gsm8k_train.jsonl"
    out_test_file = "/mnt/cache/luzimu/rlhf_math/data/math_gsm8k_lce_dpo_it2/data/test/math_gsm8k_test.jsonl"
    get_chosen_rejected_alignment_lce(in_files, out_train_file, out_test_file)
    
def main_count():
    in_file = "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mistral_lce_alignment_sample/math_train_lce_round7.jsonl"
    correct_num = 2
    wrong_num = 2
    count_valid_num(in_file, correct_num, wrong_num)
    
def main_multi():
    in_files = ["/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mistral_lce_alignment_sample/gsm8k_train_lce_round53.jsonl",
                "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mistral_lce_alignment_sample/math_train_lce_round7.jsonl"]
    num_correct = 3
    num_wrong = 3
    out_dir = f"/mnt/cache/luzimu/rlhf_math/data/math_gsm8k_lce_dpo_multi_{num_correct}x{num_wrong}"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/math_gsm8k_test.jsonl"
    get_chosen_rejected_alignment_lce_multi(in_files, out_train_file, out_test_file, num_correct, num_wrong)

def main_multi_file_diff():
    in_files_and_num = [("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mistral_lce_alignment_sample/gsm8k_train_lce_round53.jsonl", 1, 1),
                ("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mistral_lce_alignment_sample/math_train_lce_round7.jsonl", 3, 3)]
    out_dir = f"/mnt/cache/luzimu/rlhf_math/data/math_gsm8k_lce_dpo_multi_1x1-3x3"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/math_gsm8k_test.jsonl"
    get_chosen_rejected_alignment_lce_multi_file_diff(in_files_and_num, out_train_file, out_test_file)
    
def main_multi_file_diff_1x3_1x3():
    in_files_and_num = [("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mistral_lce_alignment_sample/gsm8k_train_lce_round53.jsonl", 1, 3),
                ("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mistral_lce_alignment_sample/math_train_lce_round7.jsonl", 1, 3)]
    out_dir = f"/mnt/cache/luzimu/rlhf_math/data/math_gsm8k_lce_dpo_multi_1x3-1x3"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/math_gsm8k_test.jsonl"
    get_chosen_rejected_alignment_lce_multi_file_diff(in_files_and_num, out_train_file, out_test_file)

def main_mixtral():
    in_files = ["/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mixtral_lce_alignment_sample/gsm8k_train_lce_round38.jsonl",
                "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mixtral_lce_alignment_sample/math_train_lce_round12.jsonl"]
    out_dir = f"/mnt/cache/luzimu/rlhf_math/data/mixtral_math_gsm8k_lce_dpo"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/math_gsm8k_test.jsonl"
    get_chosen_rejected_alignment_lce(in_files, out_train_file, out_test_file)
    
def main_mixtral_1x3_1x3():
    in_files_and_num = [("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mixtral_lce_alignment_sample/gsm8k_train_lce_round38.jsonl", 1, 3),
                ("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mixtral_lce_alignment_sample/math_train_lce_round12.jsonl", 1, 3)]
    out_dir = f"/mnt/cache/luzimu/rlhf_math/data/mixtral_math_gsm8k_lce_dpo_multi_1x3-1x3"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/math_gsm8k_test.jsonl"
    get_chosen_rejected_alignment_lce_multi_file_diff(in_files_and_num, out_train_file, out_test_file)

def main_mistral_mixtral_1x3_1x3():
    in_files_and_num = [("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mixtral_lce_alignment_sample/gsm8k_train_lce_round38.jsonl", 1, 3),
                ("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mixtral_lce_alignment_sample/math_train_lce_round12.jsonl", 1, 3),
                ("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mistral_lce_alignment_sample/gsm8k_train_lce_round53.jsonl", 1, 3),
                ("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mistral_lce_alignment_sample/math_train_lce_round7.jsonl", 1, 3),]
    out_dir = f"/mnt/cache/luzimu/rlhf_math/data/mistral_mixtral_math_gsm8k_lce_dpo_multi_1x3-1x3"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/math_gsm8k_test.jsonl"
    get_chosen_rejected_alignment_lce_multi_file_diff(in_files_and_num, out_train_file, out_test_file)

def main_multi_file_diff_1x3_1x3_prefer_long():
    in_files_and_num = [("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mistral_lce_alignment_sample/gsm8k_train_lce_round53.jsonl", 1, 3),
                ("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mistral_lce_alignment_sample/math_train_lce_round7.jsonl", 1, 3)]
    out_dir = f"/mnt/cache/luzimu/rlhf_math/data/math_gsm8k_lce_dpo_multi_1x3-1x3_prefer_long"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/math_gsm8k_test.jsonl"
    get_chosen_rejected_alignment_lce_multi_file_diff_prefer_long(in_files_and_num, out_train_file, out_test_file)

def main_ablation():
    for i in range(1, 5):
        for j in range(1, 5):
            print(f"{i}, {j}:")
            in_files_and_num = [("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mistral_lce_alignment_sample_1/processed_results/gsm8k_results_7473.jsonl", i, j),
                ("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mistral_lce_alignment_sample_1/processed_results/math_results_7500.jsonl", i, j)]
            out_dir = f"/mnt/cache/luzimu/rlhf_math/data/math_gsm8k_lce_dpo_{i}x{j}-{i}x{j}"
            if not os.path.exists(f"{out_dir}/data/train/"):
                os.makedirs(f"{out_dir}/data/train/")
            if not os.path.exists(f"{out_dir}/data/test/"):
                os.makedirs(f"{out_dir}/data/test/")
            out_train_file = f"{out_dir}/data/train/math_gsm8k_train.jsonl"
            out_test_file = f"{out_dir}/data/test/math_gsm8k_test.jsonl"
            get_chosen_rejected_alignment_lce_multi_file_diff(in_files_and_num, out_train_file, out_test_file)

def main_1x1_1x2():
    in_files_and_num = [("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mistral_lce_alignment_sample_1/processed_results/gsm8k_results_7473.jsonl", 1, 1),
        ("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mistral_lce_alignment_sample_1/processed_results/math_results_7500.jsonl", 1, 2)]
    out_dir = f"/mnt/cache/luzimu/rlhf_math/data/math_gsm8k_lce_dpo_1x1-1x2"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/math_gsm8k_test.jsonl"
    get_chosen_rejected_alignment_lce_multi_file_diff(in_files_and_num, out_train_file, out_test_file)


def main_1x2_1x3():
    in_files_and_num = [("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mistral_lce_alignment_sample_1/processed_results/gsm8k_results_7473.jsonl", 1, 2),
        ("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mistral_lce_alignment_sample_1/processed_results/math_results_7500.jsonl", 1, 3)]
    out_dir = f"/mnt/cache/luzimu/rlhf_math/data/math_gsm8k_lce_dpo_1x2-1x3"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/math_gsm8k_test.jsonl"
    get_chosen_rejected_alignment_lce_multi_file_diff(in_files_and_num, out_train_file, out_test_file)

def main_internlm_1x1_1x4_1x1():
    in_files_and_num = [("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/gsm8k_results_7473.jsonl", 1, 1),
        ("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/math_results_7500.jsonl", 1, 4),
        ("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/ape_results_200488.jsonl", 1, 1),]
    out_dir = f"/mnt/cache/luzimu/rlhf_math/data/internlm_gsm8k_math_ape_1x1_1x4_1x1"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/math_gsm8k_test.jsonl"
    get_chosen_rejected_alignment_lce_multi_file_diff(in_files_and_num, out_train_file, out_test_file)
    
def main_internlm_1x1_1x4():
    in_files_and_num = [("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/gsm8k_results_7473.jsonl", 1, 1),
        ("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/math_results_7500.jsonl", 1, 4)]
    out_dir = f"/mnt/cache/luzimu/rlhf_math/data/internlm_gsm8k_math_1x1_1x4"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/math_gsm8k_test.jsonl"
    get_chosen_rejected_alignment_lce_multi_file_diff(in_files_and_num, out_train_file, out_test_file)

def main_internlm_1x1_1x4_no_ch():
    in_files_and_num = [("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_no_ch_negative_positive_gen/naive_dpo/processed_results/gsm8k_results_7473.jsonl", 1, 1),
        ("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_no_ch_negative_positive_gen/naive_dpo/processed_results/math_results_7500.jsonl", 1, 4)]
    out_dir = f"/mnt/cache/luzimu/rlhf_math/data/internlm_gsm8k_math_no_ch_1x1_1x4"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/math_gsm8k_test.jsonl"
    get_chosen_rejected_alignment_lce_multi_file_diff(in_files_and_num, out_train_file, out_test_file)

def main_internlm_1x1_1x3_no_ch():
    in_files_and_num = [("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_no_ch_negative_positive_gen/naive_dpo/processed_results/gsm8k_results_7473.jsonl", 1, 1),
        ("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_no_ch_negative_positive_gen/naive_dpo/processed_results/math_results_7500.jsonl", 1, 3)]
    out_dir = f"/mnt/cache/luzimu/rlhf_math/data/internlm_gsm8k_math_no_ch_1x1_1x3"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/math_gsm8k_test.jsonl"
    get_chosen_rejected_alignment_lce_multi_file_diff(in_files_and_num, out_train_file, out_test_file)
    
def main_internlm_1x1_1x4_1x1_quarterape():
    in_files_and_num = [("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/gsm8k_results_7473.jsonl", 1, 1),
        ("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/math_results_7500.jsonl", 1, 4),
        ("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/ape_results_200488.jsonl", 1, 1),]
    out_dir = f"/mnt/cache/luzimu/rlhf_math/data/internlm_gsm8k_math_ape_1x1_1x4_1x1_quarterape"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/math_gsm8k_test.jsonl"
    get_chosen_rejected_alignment_lce_multi_file_diff(in_files_and_num, out_train_file, out_test_file, (1, 1, 0.25))
    
def main_internlm_1x1_1x4_1x1_halfape():
    in_files_and_num = [("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/gsm8k_results_7473.jsonl", 1, 1),
        ("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/math_results_7500.jsonl", 1, 4),
        ("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/ape_results_200488.jsonl", 1, 1),]
    out_dir = f"/mnt/cache/luzimu/rlhf_math/data/internlm_gsm8k_math_ape_1x1_1x4_1x1_halfape"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/math_gsm8k_test.jsonl"
    get_chosen_rejected_alignment_lce_multi_file_diff(in_files_and_num, out_train_file, out_test_file, (1, 1, 0.5))
    
def main_internlm_1x1_1x1_1x1_quarterape():
    in_files_and_num = [("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/gsm8k_results_7473.jsonl", 1, 1),
        ("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/math_results_7500.jsonl", 1, 1),
        ("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/ape_results_200488.jsonl", 1, 1),]
    out_dir = f"/mnt/cache/luzimu/rlhf_math/data/internlm_gsm8k_math_ape_1x1_1x1_1x1_quarterape"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/math_gsm8k_test.jsonl"
    get_chosen_rejected_alignment_lce_multi_file_diff(in_files_and_num, out_train_file, out_test_file, (1, 1, 0.25))
    
def main_internlm_1x2_1x7_1x1_quarterape():
    in_files_and_num = [(["/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/gsm8k_results_7473.jsonl",
                          "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_no_ch_negative_positive_gen/naive_dpo/processed_results/gsm8k_results_7473.jsonl"], 1, 2),
        (["/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/math_results_7500.jsonl",
          "/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_no_ch_negative_positive_gen/naive_dpo/processed_results/math_results_7500.jsonl"], 1, 7),
        (["/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/ape_results_200488.jsonl"], 1, 1),]
    out_dir = f"/mnt/cache/luzimu/rlhf_math/data/internlm_gsm8k_math_ape_1x2_1x7_1x1_0.35ape"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/math_gsm8k_test.jsonl"
    get_chosen_rejected_alignment_lce_multi_file_diff_add_neg_from_other(in_files_and_num, out_train_file, out_test_file, (1, 1, 0.35))

def main_mathcoder_1x1_1x1():
    in_files_and_num = [("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mathcoder_dpo/naive_dpo/processed_results/gsm8k_results_7473.jsonl", 1, 1),
        ("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mathcoder_dpo/naive_dpo/processed_results/math_results_7500.jsonl", 1, 1),]
    out_dir = f"/mnt/cache/luzimu/rlhf_math/data/mathcoder_gsm8k_math_ape_1x1_1x1"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/math_gsm8k_test.jsonl"
    get_chosen_rejected_alignment_lce_multi_file_diff(in_files_and_num, out_train_file, out_test_file, (1, 1))

def main_mathcoder_1x1_1x1_addsys():
    in_files_and_num = [("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mathcoder_dpo/naive_dpo/processed_results/gsm8k_results_7473.jsonl", 1, 1),
        ("/mnt/cache/luzimu/rlhf_math/data/lce_solutions/mathcoder_dpo/naive_dpo/processed_results/math_results_7500.jsonl", 1, 1),]
    out_dir = f"/mnt/cache/luzimu/rlhf_math/data/mathcoder_gsm8k_math_ape_1x1_1x1"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/math_gsm8k_test.jsonl"
    get_chosen_rejected_alignment_lce_multi_file_diff(in_files_and_num, out_train_file, out_test_file, (1, 1))

if __name__ == "__main__":
    main_mathcoder_1x1_1x1_addsys()