import json
import os
import sys
from tqdm import tqdm
from argparse import ArgumentParser

sys.path.append("src/step_controled_dpo_lce_internlm")
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

def main():
    args = get_args()
    in_file = f'data/lce_solutions/internlm_negative_positive_gen/sc_dpo/gsm8k/{args.i}_round{args.r}.jsonl'
    source_file = f'data/lce_solutions/internlm_negative_positive_gen/sc_dpo/gsm8k/to_be_run_{args.i}_round{args.r}.jsonl'
    datas = load_jsonl(in_file)
    if len(datas) < len(load_jsonl(source_file)):
        raise ValueError(f"Running index{args.i} round{args.r} not finished")
    
    wrong_datas = []
    no_wrong_datas = []
    for data in tqdm(datas):
        if not is_equal(data["debug_result"][-1]["content"], data["extra"]["answer"]):
            wrong_datas.append(data)
        else:
            data.pop("debug_result", None)
            no_wrong_datas.append(data)
            
    out_file_to_be_run = f'data/lce_solutions/internlm_negative_positive_gen/sc_dpo/gsm8k/to_be_run_{args.i}_round{args.r + 1}.jsonl'
    out_file_result = f'data/lce_solutions/internlm_negative_positive_gen/sc_dpo/gsm8k/result_{args.i}_round{args.r}.jsonl'
    save_jsonl(wrong_datas, out_file_result)
    save_jsonl(no_wrong_datas, out_file_to_be_run)

if __name__ == "__main__":
    main()