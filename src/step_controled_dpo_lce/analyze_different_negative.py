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

def analyze_different_negative(in_file, out_file):
    datas = load_jsonl(in_file)
    
    num_of_negative_dict = {} # dict of num_of_negative_solutions: number_of_question_with_this_num
    start_steps_dict = {} # dict of start_steps: number_of_negative_solutions_augmented_at_this_start_step
    total_num = 0
    total_wrong_solution_num = 0
    total_start_steps_num = 0
    
    for data in tqdm(datas):
        if len(data["correct_solution"]) > 0:
            total_num += 1
            num_negative = len(data["wrong_solutions"])
            if num_negative in num_of_negative_dict.keys():
                num_of_negative_dict[num_negative] += 1
            else:
                num_of_negative_dict[num_negative] = 1
            for wrong_solution in data["wrong_solutions"]:
                total_wrong_solution_num += 1
                start_steps = wrong_solution["start_steps"]
                total_start_steps_num += start_steps
                if start_steps in start_steps_dict.keys():
                    start_steps_dict[start_steps] += 1
                else:
                    start_steps_dict[start_steps] = 1
                    
    result_dict = {
        "num_of_negative_dict": num_of_negative_dict,
        "start_steps_dict": start_steps_dict,
        "total_num": total_num,
        "total_wrong_solution_num": total_wrong_solution_num,
        "average_wrong_solution_num": total_wrong_solution_num / total_num,
        "average_start_step": total_start_steps_num / total_wrong_solution_num
    }
    
    with open(out_file, "w") as f:
        json.dump(result_dict, f)
        

def main_gsm8k():
    in_file = "data/lce_solutions/different_ranked_negative_divided/processed_results/gsm8k_train_lce_round53_step_controled_negative.jsonl"
    out_file = "data/lce_solutions/different_ranked_negative_divided/processed_results/analyze_gsm8k_train_lce_round53_step_controled_negative.json"
    analyze_different_negative(in_file, out_file)
    
def main_math():
    in_file = "data/lce_solutions/different_ranked_negative_divided/processed_results/math_train_lce_round7_step_controled_negative.jsonl"
    out_file = "data/lce_solutions/different_ranked_negative_divided/processed_results/analyze_math_train_lce_round7_step_controled_negative.json"
    analyze_different_negative(in_file, out_file)

def main_gsm8k_tmp1():
    in_file = "data/lce_solutions/different_ranked_negative_divided_tmp1/processed_results/gsm8k_train_lce_round53_step_controled_negative.jsonl"
    out_file = "data/lce_solutions/different_ranked_negative_divided_tmp1/processed_results/analyze_gsm8k_train_lce_round53_step_controled_negative.json"
    analyze_different_negative(in_file, out_file)
    
def main_math_tmp1():
    in_file = "data/lce_solutions/different_ranked_negative_divided_tmp1/processed_results/math_train_lce_round7_step_controled_negative.jsonl"
    out_file = "data/lce_solutions/different_ranked_negative_divided_tmp1/processed_results/analyze_math_train_lce_round7_step_controled_negative.json"
    analyze_different_negative(in_file, out_file)
    
def main_gsm8k_ascend():
    in_file = "data/lce_solutions/ascending_temperature_negative/processed_results/gsm8k_train_lce_round53_step_controled_negative.jsonl"
    out_file = "data/lce_solutions/ascending_temperature_negative/processed_results/analyze_gsm8k_train_lce_round53_step_controled_negative.json"
    analyze_different_negative(in_file, out_file)
    
def main_math_ascend():
    in_file = "data/lce_solutions/ascending_temperature_negative/processed_results/math_train_lce_round7_step_controled_negative.jsonl"
    out_file = "data/lce_solutions/ascending_temperature_negative/processed_results/analyze_math_train_lce_round7_step_controled_negative.json"
    analyze_different_negative(in_file, out_file)
    
if __name__ == "__main__":
    main_gsm8k_ascend()
    main_math_ascend()