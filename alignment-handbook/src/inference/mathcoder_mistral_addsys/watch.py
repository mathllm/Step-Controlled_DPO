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
    chs = ["2epoch"]
    count = {}
    for ch in chs:
        count[ch] = {"GSM8K": 0, "MATH_0": 0, "MATH_1": 0, "MATH_2": 0, "MATH_3": 0, "SVAMP": 0, "simuleq": 0, "mathematics": 0, "asdiv": 0, "mawps": 0}
    while True:
        for ch in chs:
            print(f"{ch}:")
            dir1 = dir + "/" + ch
            for name in ["GSM8K", "MATH_0", "MATH_1", "MATH_2", "MATH_3", "SVAMP", "simuleq", "mathematics", "asdiv", "mawps"]:
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
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, "config.json"), "r") as f:
        config = json.load(f)
    
    dir = f"{config['model_name']}/"
    watch(dir)