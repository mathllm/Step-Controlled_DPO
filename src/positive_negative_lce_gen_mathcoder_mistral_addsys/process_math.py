import json
import os
import sys
from tqdm import tqdm
from argparse import ArgumentParser

sys.path.append("src/different_negative_gen")
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

def get_string_from_solution(debug_result):
    text = ""
    for block in debug_result:
        text += block["content"]
    return text

def no_similar(debug_result, solutions):
    for solution in solutions:
        if len(debug_result) == len(solution) and abs(len(get_string_from_solution(debug_result)) - len(get_string_from_solution(solution))) < 500:
            return False
    return True

def exist_error(solution):
    solution = solution.lower()
    error_phrases = ["error", "apolog"]
    for error_phrase in error_phrases:
        if error_phrase in solution:
            return True
    return False

def process(args, in_file, source_file, result_file, to_be_run_file, num_correct_thresh, num_wrong_thresh):
    datas = load_jsonl(in_file)
    result_datas = load_jsonl(result_file)
    to_be_run_datas = []
    if len(datas) < len(load_jsonl(source_file)):
        raise ValueError(f"Running index{args.i} round{args.r} not finished")

    for data in tqdm(datas):
        idx = data["idx"]
        new_data = {
            "question": data["question"],
            "extra": data["extra"],
            "correct_num": data["correct_num"],
            "wrong_num": data["wrong_num"],
            "idx": data["idx"]
        }
        if "correct_solutions" not in result_datas[idx].keys():
            result_datas[idx]["correct_solutions"] = []
        if "wrong_solutions" not in result_datas[idx].keys():
            result_datas[idx]["wrong_solutions"] = []
        if "correct_errored_solutions" not in result_datas[idx].keys():
            result_datas[idx]["correct_errored_solutions"] = []
        if is_equal(data["debug_result"][-1]["content"], data["extra"]["answer"]):
            if exist_error(get_string_from_solution(data["debug_result"])):
                result_datas[idx]["correct_errored_solutions"].append(data["debug_result"])
            elif no_similar(data["debug_result"], result_datas[idx]["correct_solutions"]):
                result_datas[idx]["correct_solutions"].append(data["debug_result"])
                new_data["correct_num"] += 1
        elif no_similar(data["debug_result"], result_datas[idx]["wrong_solutions"]):
            result_datas[idx]["wrong_solutions"].append(data["debug_result"])
            new_data["wrong_num"] += 1

        if new_data["correct_num"] < num_correct_thresh or new_data["wrong_num"] < num_wrong_thresh:
            to_be_run_datas.append(new_data)
    save_jsonl(result_datas, result_file)
    save_jsonl(to_be_run_datas, to_be_run_file)

def main():
    args = get_args()
    num_correct_thresh = 1
    num_wrong_thresh = 4
    in_file = f'data/lce_solutions/mathcoder_mistral_dpo_addsys/naive_dpo/math/{args.i}_round{args.r}.jsonl'
    source_file = f'data/lce_solutions/mathcoder_mistral_dpo_addsys/naive_dpo/math/to_be_run_{args.i}_round{args.r}.jsonl'
    result_file = f"data/lce_solutions/mathcoder_mistral_dpo_addsys/naive_dpo/math/result_{args.i}.jsonl"
    to_be_run_file = f'data/lce_solutions/mathcoder_mistral_dpo_addsys/naive_dpo/math/to_be_run_{args.i}_round{args.r + 1}.jsonl'
    process(args, in_file, source_file, result_file, to_be_run_file, num_correct_thresh, num_wrong_thresh)
            

if __name__ == "__main__":
    main()
