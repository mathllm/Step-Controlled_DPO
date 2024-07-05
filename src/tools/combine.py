import os
from random import shuffle, seed
import json
from tqdm import tqdm

def save_jsonl(datas, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        for data in datas:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

def load_json(in_file):
    datas = []
    with open(in_file, "r", encoding="utf-8") as f:
        # datas = [json.loads(line) for line in f]
        for line in tqdm(f):
            try:
                datas.append(json.loads(line))
            except:
                print(line)
    return datas

def save_ddp(out_file, world_size):
    for i in range(world_size):
        if i == 0:
            os.system('cat %s > %s' % (out_file + '.%d' % i, out_file))
        else:
            os.system('cat %s >> %s' % (out_file + '.%d' % i, out_file))
        os.system('rm %s' % (out_file + '.%d' % i))
        
def combine(in_files, out_file):
    seed(3407)
    new_datas = []
    for in_file in in_files:
        print(in_file)
        datas = load_json(in_file)
        for data in datas:
            new_datas.append(data)
    
    shuffle(new_datas)
    print(len(new_datas))
    save_jsonl(new_datas, out_file[:-6] + f"_{len(new_datas)}.jsonl")
    # save_jsonl(new_datas, out_file)
    
def combine_with_weight(in_files, out_file):
    seed(3407)
    new_datas = []
    for in_file, w in in_files:
        print(in_file)
        datas = load_json(in_file)
        shuffle(datas)
        n = int(w * len(datas))
        for data in tqdm(datas[:n]):
            messages = data["messages"]
            if messages[0]["role"] != "system":
                messages = [{"role": "system", "content": [{"type": "text", "content": ""}]},] + messages
            new_datas.append({"messages":messages})
    
    shuffle(new_datas)
    print(len(new_datas))
    save_jsonl(new_datas, out_file[:-6] + f"_{len(new_datas)}.jsonl")
    # save_jsonl(new_datas, out_file)
    
def main_gsm8k():
    in_files = [f"data/lce_solutions/mistral_lce_alignment_sample_1/gsm8k/result_{i}.jsonl" for i in range(6)]
    out_file = "data/lce_solutions/mistral_lce_alignment_sample_1/processed_results/gsm8k_results.jsonl"
    combine(in_files, out_file)

def main_math():
    in_files = [f"data/lce_solutions/mistral_lce_alignment_sample_1/math/result_{i}.jsonl" for i in range(6)]
    out_file = "data/lce_solutions/mistral_lce_alignment_sample_1/processed_results/math_results.jsonl"
    combine(in_files, out_file)

def main_gsm8k_internlm():
    in_files = [f"data/lce_solutions/internlm_negative_positive_gen/naive_dpo/gsm8k/result_{i}.jsonl" for i in range(12)]
    out_file = "data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/gsm8k_results.jsonl"
    combine(in_files, out_file)

def main_math_internlm():
    in_files = [f"data/lce_solutions/internlm_negative_positive_gen/naive_dpo/math/result_{i}.jsonl" for i in range(12)]
    out_file = "data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/math_results.jsonl"
    combine(in_files, out_file)

def main_ape_internlm():
    in_files = [f"data/lce_solutions/internlm_negative_positive_gen/naive_dpo/ape/result_{i}.jsonl" for i in range(12)]
    out_file = "data/lce_solutions/internlm_negative_positive_gen/naive_dpo/processed_results/ape_results.jsonl"
    combine(in_files, out_file)
    
def main_gsm8k_internlm_no_ch():
    in_files = [f"data/lce_solutions/internlm_no_ch_negative_positive_gen/naive_dpo/gsm8k/result_{i}.jsonl" for i in range(18)]
    out_file = "data/lce_solutions/internlm_no_ch_negative_positive_gen/naive_dpo/processed_results/gsm8k_results.jsonl"
    combine(in_files, out_file)

def main_math_internlm_no_ch():
    in_files = [f"data/lce_solutions/internlm_no_ch_negative_positive_gen/naive_dpo/math/result_{i}.jsonl" for i in range(18)]
    out_file = "data/lce_solutions/internlm_no_ch_negative_positive_gen/naive_dpo/processed_results/math_results.jsonl"
    combine(in_files, out_file)
    
def main_gsm8k_mathcoder():
    in_files = [f"data/lce_solutions/mathcoder_dpo/naive_dpo/gsm8k/result_{i}.jsonl" for i in range(10)]
    out_file = "data/lce_solutions/mathcoder_dpo/naive_dpo/processed_results/gsm8k_results.jsonl"
    combine(in_files, out_file)

def main_math_mathcoder():
    in_files = [f"data/lce_solutions/mathcoder_dpo/naive_dpo/math/result_{i}.jsonl" for i in range(20)]
    out_file = "data/lce_solutions/mathcoder_dpo/naive_dpo/processed_results/math_results.jsonl"
    combine(in_files, out_file)
    
def main_gsm8k_mathcoder_addsys():
    in_files = [f"data/lce_solutions/mathcoder_dpo_addsy/naive_dpo/gsm8k/result_{i}.jsonl" for i in range(16)]
    out_file = "data/lce_solutions/mathcoder_dpo_addsy/naive_dpo/processed_results/gsm8k_results.jsonl"
    combine(in_files, out_file)

def main_math_mathcoder_addsys():
    in_files = [f"data/lce_solutions/mathcoder_dpo_addsy/naive_dpo/math/result_{i}.jsonl" for i in range(16)]
    out_file = "data/lce_solutions/mathcoder_dpo_addsy/naive_dpo/processed_results/math_results.jsonl"
    combine(in_files, out_file)

if __name__ == "__main__":
    main_gsm8k_mathcoder_addsys()
    main_math_mathcoder_addsys()