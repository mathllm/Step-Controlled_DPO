import json
import os
import re
from tqdm import tqdm
import time
from argparse import ArgumentParser
from random import shuffle, seed

def save_jsonl(datas, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        for data in datas:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

def load_jsonl(in_file):
    with open(in_file, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]
    return datas

def sample(in_files, out_file):
    new_datas = []
    
    for in_file in in_files:
        datas = load_jsonl(in_file)
    
        seed(3407)
        shuffle(datas)
        new_datas.extend(datas[:50])
    
    shuffle(new_datas)
    
    save_jsonl(new_datas, out_file)
    
def get_md(in_file, out_file):
    datas1 = load_jsonl(in_file)
    
    datas_list = [
        datas1[0:25],
        datas1[25:50],
        datas1[50:75],
        datas1[75:100]
    ]
    
    cnt = 0
    cnt_steps = 0
    for idx, datas in enumerate(datas_list):
        with open(out_file[:-3] + f"_{idx}.md", "w") as f:
            for idx, data in enumerate(datas):
                f.write("*" * 99 + f"\nIndex:{idx}\n\n")
                f.write(f"Question:{data['extra']['question']}\n\n")
                f.write(f"GT Solution:{data['extra']['solution']}\n\n")
                f.write(f"GT Answer:{data['extra']['answer']}\n\n")
                f.write("*" * 99 + "\n\n")
                if len(data["correct_solution"]) > 0:
                    cnt += 1
                for block in data["correct_solution"][2:-1]:
                    if block['role'] == "code" or block['role'] == "text":
                        f.write("<|step|>\n\n")
                        cnt_steps += 1
                    f.write(f"{block['role']}:\n\n{block['content']}\n\n")
                    
    print(cnt)
    print(cnt_steps)
        
    
def main():
    in_files = ["data/lce_solutions/ascending_temperature_negative/processed_results/gsm8k_train_lce_round53_step_controled_negative.jsonl",
                "data/lce_solutions/ascending_temperature_negative/processed_results/math_train_lce_round7_step_controled_negative.jsonl"]
    out_file = "src/tools/sample/50_gsm8k_50_math.jsonl"
    sample(in_files, out_file)
    
def main_md():
    in_file = "src/tools/sample/50_gsm8k_50_math.jsonl"
    out_file = "src/tools/sample/50_gsm8k_50_math.md"
    get_md(in_file, out_file)
    
if __name__ == "__main__":
    main()
    main_md()