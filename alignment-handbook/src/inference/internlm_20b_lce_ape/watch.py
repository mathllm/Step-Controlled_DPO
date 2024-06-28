import json
import os
import re
from tqdm import tqdm
import time
from argparse import ArgumentParser

def save_jsonl(datas, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        for data in datas:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

def load_jsonl(in_file):
    with open(in_file, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]
    return datas

def watch(dir):
    chs = ["3epoch"]
    count = {}
    for ch in chs:
        count[ch] = {"GSM8K": 0, "MATH_0": 0, "MATH_1": 0, "MATH_2": 0, "MATH_3": 0, "SVAMP": 0, "simuleq": 0, "mathematics": 0, "APE_0": 0, "APE_1": 0, "APE_2": 0, "APE_3": 0, "cmath": 0, "mgsm_zh": 0}
    while True:
        for ch in chs:
            print(f"{ch}:")
            dir1 = dir + "/" + ch
            for name in count[chs[0]].keys():
                source_file = f'/mnt/cache/luzimu/open_source_repositories/Step-Controlled_DPO/alignment-handbook/src/inference/all_test/{name}_test.jsonl'
                file_path = f'/mnt/cache/luzimu/rlhf_math/alignment-handbook/results/inference/{dir1}/{name}/{name}_test_result.jsonl'

                if not os.path.exists(file_path):
                    length = 0
                else:
                    length = len(load_jsonl(file_path))

                total_length = len(load_jsonl(source_file))
                if length == count[ch][name]:
                    print(f"{name} no change: {length} ({total_length-length})")
                else:
                    print(f"{name}: {length} ({total_length-length})")
                    count[ch][name] = length
            print("---------------------------------")

        print("\n***************************************\n")
        time.sleep(600)

if __name__ == "__main__":
    # # parser = ArgumentParser(description="A simple argument parser")
    # # parser.add_argument("ch", type=str, help="checkpoint_number", default="600")
    # args = parser.parse_args()
    dir = "internlm2-20b_ape_th1_169161_gsm8k_math_81087/sft/"
    watch(dir)