import json
import os
import sys
from tqdm import tqdm
from argparse import ArgumentParser

sys.path.append("/mnt/cache/luzimu/rlhf_math/src/step_controled_dpo_lce_internlm")
from utils import is_equal

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
    parser.add_argument("-i", type=str, help="index")
    args = parser.parse_args()
    return args

def infer_error(debug_result):
    for block in debug_result:
        if block["role"] == "exceed_max_length/return_first_code":
            return True
    return False

def main():
    args = get_args()
    in_file = f'/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/sc_dpo/gsm8k/{args.i}_round{args.r}.jsonl'
    source_file = f'/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/sc_dpo/gsm8k/to_be_run_{args.i}_round{args.r}.jsonl'
    datas = load_jsonl(in_file)
    if len(datas) < len(load_jsonl(source_file)):
        raise ValueError(f"Running index{args.i} round{args.r} not finished")
    
    wrong_datas = []
    no_wrong_datas = []
    for data in tqdm(datas):
        if not is_equal(data["debug_result"][-1]["content"], data["extra"]["answer"]) and not infer_error(data["debug_result"]):
            wrong_datas.append(data)
        elif data["start_steps"] > 0:
            data.pop("debug_result", None)
            no_wrong_datas.append(data)
            
    out_file_to_be_run = f'/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/sc_dpo/gsm8k/to_be_run_{args.i}_round{args.r + 1}.jsonl'
    out_file_result = f'/mnt/cache/luzimu/rlhf_math/data/lce_solutions/internlm_negative_positive_gen/sc_dpo/gsm8k/result_{args.i}_round{args.r}.jsonl'
    save_jsonl(wrong_datas, out_file_result)
    save_jsonl(no_wrong_datas, out_file_to_be_run)

if __name__ == "__main__":
    main()