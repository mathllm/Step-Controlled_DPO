import json
import os
import re
from latex2sympy2 import latex2sympy
from tqdm import tqdm
from argparse import ArgumentParser

def save_jsonl(datas, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        for data in datas:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

def load_json(in_file):
    with open(in_file, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]
    return datas

def find_all_numbers1(text):
    last_line = text.split("\n\n")[-1]
    # The regex pattern to match any number (integer or floating-point)
    pattern = r'-?\d+(?:\.\d+)?'
    
    # Using findall to get all occurrences of numbers in the string
    all_numbers = re.findall(pattern, text)
    
    # If there are no numbers in the string, return None
    if not all_numbers:
        return None
    
    # Return the last number found in the string
    return all_numbers

def find_all_numbers(text):
    # last_line = text.split("\n\n")[-1]
    pattern = re.compile('oxed{(.*)}',flags=re.S)
    answers = pattern.findall(text)
    for i in range(len(answers)):
        answers[i] = answers[i].replace(",", "")

    all_numbers = []
    for answer in answers:
        # The regex pattern to match any number (integer or floating-point)
        pattern = r'-?\d+(?:\.\d+)?'
        
        # Using findall to get all occurrences of numbers in the string
        all_numbers.extend(re.findall(pattern, answer))
    
    # If there are no numbers in the string, return None
    if not all_numbers:
        return None
    
    # Return the last number found in the string
    return all_numbers

def delete_extra_zero(n):
    '''删除小数点后多余的0'''
    try:
        n=float(n)
    except:
        print("None {}".format(n))
        return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip('0')  # 删除小数点后多余的0
        n = int(n.rstrip('.')) if n.endswith('.') else float(n)  # 只剩小数点直接转int，否则转回float
        n=str(n)
        return n


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    splits = string.split("\\text{ ")
    # assert len(splits) == 2
    return splits[-1]


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if len(split) > 0 and split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

    
def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        string = string.split("=")[-1]
    if len(string.split("\\approx")) == 2:
        string = string.split("\\approx")[-1]

    # fix sqrt3 --> sqrt{3}
    if 'sqrt' in string:
        string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    if 'sqrt' in string:
        string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

    
def find_math_answer(s:str, type="gpt"):
    s = s.lower()
    if '{}' in s:
        s = s.replace('{}','')
    try:
        pattern = re.compile('oxed{(.*)}',flags=re.S)
        ans = pattern.findall(s)[-1]
    except:
        if type == "gpt":
            ans = ""
        else:
            ans = s
 
    if ans.find('}') != -1 and(ans.find('{') ==-1 or  ans.find('}') < ans.find('{')):
        ans=ans.split('}')[0]
    # remove
    ans = ans.split('=')[-1]
    ans = ans.split('\\approx')[-1]
    ans = ans.replace(" ", "")
    ans = ans.replace("\\,", "")
    ans = ans.replace('∞','\\infty').replace("+\infty", "\infty")
    ans = ans.replace("\\\\", "\\")
    ans = ans.replace("\n", "")
    ans = ans.replace('\\text', '').replace('\\mbox', '')
    ans = ans.replace('bmatrix', 'pmatrix')
    ans = ans.replace("\\left", "").replace('\\right', '')
    ans = ans.replace("^{\\circ}", "").replace("^\\circ", "")
    ans = ans.replace("{m}^3", "").replace("m^3", "")
    ans = ans.replace("{units}", "").replace("units", "")
    ans = ans.replace("{km}", "").replace("km", "")
    return _strip_string(ans)


def eval_tuple(s):
    """
    (a,b,c,...)
    """
    sl = s[1:-1].split(',')
    try :
        if s[0] == '(' and s[-1] == ')' and len(sl) > 1:
            s = ','.join([str(round(eval(str(latex2sympy(sub))),2)) if 'infty' not in sub and sub not in ['a', '-a'] else sub for sub in sl])
            return f"({s})"
    except:
        return s
    return s


def is_equal(asw:str, gt_asw:str) -> bool:
    """
    Judge if asw is equivalent to gt_asw.
    """
    asw = find_math_answer(asw, "gt")
    gt_asw = find_math_answer(gt_asw, "gt")
    if gt_asw == "" or asw == "":
        return False
    asw = eval_tuple(asw)
    gt_asw = eval_tuple(gt_asw)
    if gt_asw == asw:
        return True
    else:
        try:
            if ',' in gt_asw:
                if set(gt_asw.split(',')) == set(asw.split(',')):
                    return True
            if str(round(eval(str(latex2sympy(gt_asw))),2)) == str(round(eval(str(latex2sympy(asw))),2)):
                return True
            
            else:
                return False
        except:
            return False


def compute_accuracy(in_file, out_path, orig_file):
    """
    compute accuracy for MATH like datasets
    with answers that are not all numbers
    """
    with open(in_file, "r") as f:
        datas = [json.loads(line) for line in f]
    
    with open(orig_file, "r") as f:
        orig_datas = [json.loads(line) for line in f]

    total_num = 0
    correct_num = 0
    new_datas = []
    wrong_datas = []
    correct_datas = []
    for data, orig_data in tqdm(zip(datas, orig_datas)):
        solution = data["completion"]
        ans = find_math_answer(solution)
        new_datas.append({"id":orig_data["id"], "solution": solution, "model_answer": ans, "answer": orig_data["answer"]})
        if ans != "" and is_equal(ans, orig_data["answer"]):
            correct_datas.append({"id":orig_data["id"], "solution": solution, "model_answer": ans, "answer": orig_data["answer"]})
            correct_num += 1
        else:
            wrong_datas.append({"id":orig_data["id"], "solution": solution, "model_answer": ans, "answer": orig_data["answer"]})
        total_num += 1

    print(f"total_acc: {correct_num / total_num}")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with open(os.path.join(out_path, "total.jsonl"), "w") as f:
        for new_data in new_datas:
            f.write(json.dumps(new_data) + "\n")

    with open(os.path.join(out_path, "correct.jsonl"), "w") as f:
        for correct_data in correct_datas:
            f.write(json.dumps(correct_data) + "\n")

    with open(os.path.join(out_path, "wrong.jsonl"), "w") as f:
        for wrong_data in wrong_datas:
            f.write(json.dumps(wrong_data) + "\n")


def extract_last_num(text: str) -> float:
    """
    extract the last number in a string
    """
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)  # 处理形如 123,456
    res = re.findall(r"(-?\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return None

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def compute_accuracy_k12(in_file, out_path):
    """
    extract last number in the last block
    and compare it to the gt answer
    """
    with open(in_file, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]

    correct_ans = []
    wrong_ans = []
    no_ans = []
    for data in datas:
        model_ans = extract_last_num(data["completion"])
        gt_ans = float(data["extra"]["answer"])
        data["model_answer"] = model_ans
        data["gt_answer"] = gt_ans
        try:
            if abs(model_ans - gt_ans) < 1e-5:
                correct_ans.append(data)
            else:
                wrong_ans.append(data)
        except:
            no_ans.append(data)

    with open(os.path.join(out_path, "correct.jsonl"), "w") as f:
        for data in correct_ans:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    with open(os.path.join(out_path, "wrong.jsonl"), "w") as f:
        for data in wrong_ans:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    with open(os.path.join(out_path, "none.jsonl"), "w") as f:
        for data in no_ans:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    print(f"correct_num: {len(correct_ans)}")
    print(f"wrong_num: {len(wrong_ans)}")
    print(f"none_num: {len(no_ans)}")
    print(f"acc: {100 * len(correct_ans) / len(datas)}")


def compute_accuracy_ours(in_file, out_file, name):
    """
    compute accuracy when the answer is put in the last block and thus in completion
    """
    with open(in_file, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]
    correct_datas = []
    wrong_datas = []
    levels = {"Level 1": {"correct": 0, "wrong": 0}, "Level 2": {"correct": 0, "wrong": 0}, "Level 3": {"correct": 0, "wrong": 0}, "Level 4": {"correct": 0, "wrong": 0}, "Level 5": {"correct": 0, "wrong": 0}}
    subjects = {"algebra": {"correct": 0, "wrong": 0}, "counting_and_probability": {"correct": 0, "wrong": 0}, "geometry": {"correct": 0, "wrong": 0}, "intermediate_algebra": {"correct": 0, "wrong": 0}, "number_theory": {"correct": 0, "wrong": 0}, "prealgebra": {"correct": 0, "wrong": 0}, "precalculus": {"correct": 0, "wrong": 0}}
    for data in tqdm(datas):
        gt_answer = data["extra"]["answer"]
        model_answer = find_math_answer(data["debug_result"][-2]["content"])
        if "completion" not in data.keys():
            data["completion"] = data["debug_result"][-1]["content"]
        if "MATH" in name:
            data["answers"] = {"gt_answer": gt_answer, "model_answer": data["completion"]}
            level = data["extra"]["level"]
            subject = data["extra"]["id"].split("/")[1]
            if is_equal(data["completion"], data["extra"]["answer"]) or ('answer_val' in data['extra'].keys() and is_equal(data["completion"], data["extra"]["answer_val"])):
            # if is_equal(data["completion"], data["ground_truth"]["answer"]):
                correct_datas.append(data)
                levels[level]["correct"] += 1
                subjects[subject]["correct"] += 1
            else:
                wrong_datas.append(data)
                levels[level]["wrong"] += 1
                subjects[subject]["wrong"] += 1
        else:
            data["answers"] = {"gt_answer": gt_answer, "model_answer": data["completion"]}
            simuleq_equal = False
            if name == "simuleq":
                if "\\boxed{" in data["debug_result"][-2]["content"]:
                    last_line = data["debug_result"][-2]["content"].split("\n\n")[-1]
                    all_numbers = find_all_numbers(last_line)
                    if all_numbers is not None and gt_answer in all_numbers:
                        simuleq_equal = True
                
            if is_equal(data["completion"], data["extra"]["answer"]) or ('answer_val' in data['extra'].keys() and is_equal(data["completion"], data["extra"]["answer_val"])) or simuleq_equal:
                correct_datas.append(data)
            else:
                wrong_datas.append(data)

    save_jsonl(correct_datas, out_file[:-6] + "_correct.jsonl")
    save_jsonl(wrong_datas, out_file[:-6] + "_wrong.jsonl")
    print(f"correct: {len(correct_datas)}")
    print(f"wrong: {len(wrong_datas)}")
    print(f"acc: {100 * len(correct_datas) / (len(correct_datas) + len(wrong_datas))}")
    
    if "MATH" in name:
        for level in [f"Level {i + 1}" for i in range(5)]:
            print(f"{level}:")
            print(f"acc {levels[level]['correct']/(levels[level]['correct']+levels[level]['wrong'])*100}")
            print(f"correct {levels[level]['correct']}")
            print(f"wrong {levels[level]['wrong']}")
        for subject in subjects.keys():
            print(f"{subject}:")
            if subjects[subject]['correct']+subjects[subject]['wrong'] > 0:
                print(f"acc {subjects[subject]['correct']/(subjects[subject]['correct']+subjects[subject]['wrong'])*100}")
            print(f"correct {subjects[subject]['correct']}")
            print(f"wrong {subjects[subject]['wrong']}")


def combine(in_dir, name="MATH"):
    MATH_files = [os.path.join(in_dir, f"{name}_{i}/{name}_{i}_test_result.jsonl") for i in range(4)]
    parts_exists = False
    for MATH_file in MATH_files:
        if os.path.exists(MATH_file):
            parts_exists = True
            break
    if parts_exists:
        os.makedirs(os.path.join(in_dir, f"{name}"), exist_ok=True)
    else:
        return
    datas = []
    for MATH_file in MATH_files:
        if not os.path.exists(MATH_file):
            continue
        datas.extend(load_json(MATH_file))
    
    save_jsonl(datas, os.path.join(in_dir, f"{name}/{name}_test_result.jsonl"))
        
def main():
    compute_accuracy_ours("/mnt/cache/luzimu/code_generation-master/data/inference/Llama-2-70b_filtered_AugGSM8K_AugMATH_ch1200_ch1600_gsm8kMath_verify_284468-2023-12-31-11:56/ch1600/vote0/simuleq/simuleq_test_result.jsonl",
    "/mnt/cache/luzimu/code_generation-master/data/inference/Llama-2-70b_filtered_AugGSM8K_AugMATH_ch1200_ch1600_gsm8kMath_verify_284468-2023-12-31-11:56/ch1600/vote0/simuleq/simuleq_test_result.jsonl",
    "simuleq")

if __name__ == "__main__":
    parser = ArgumentParser(description="argument parser")
    parser.add_argument("ch", type=str)
    parser.add_argument("-i", type=str, help="index", default="")
    args = parser.parse_args()


    dir = f"internlm2-20b_ape_th1_169161_gsm8k_math_81087/sft/{args.ch}"
    combine(f"alignment-handbook/results/inference/{dir}", "MATH")
    combine(f"alignment-handbook/results/inference/{dir}", "APE")
    # compute_accuracy("/mnt/cache/luzimu/code_generation-master/outs/Llama-2-7b-hf-math-2023-08-27-09:58/MATH_test_result.jsonl", "/mnt/cache/luzimu/code_generation-master/outs/Llama-2-7b-hf-math-2023-08-27-09:58/MATH_results", "/mnt/cache/luzimu/gsm8k-rft-llama7b-u13b_evaluation/MATH_test_orig.jsonl")
    # compute_accuracy_k12("/mnt/cache/luzimu/code_generation-master/outs/Llama-2-7b-hf-k12-2023-08-28-17:38/4200_test/k12_test_result.jsonl", "/mnt/cache/luzimu/code_generation-master/outs/Llama-2-7b-hf-k12-2023-08-28-17:38/4200_test/")
    # "TAL500", "CMMLU", "AGI", "CEval"
    # for name in ["GSM8K200", "APE500", "MATH500", "MATH"]:
    for name in ["GSM8K", "MATH", "SVAMP", "simuleq", "mathematics", "APE", "cmath", "mgsm_zh"]:
        print(name + ":")
        compute_accuracy_ours(f"alignment-handbook/results/inference/{dir}/{name}/{name}_test_result{args.i}.jsonl",
        f"alignment-handbook/results/inference/{dir}/{name}/{name}_test_result.jsonl",
        name)
    # main()


    
    #############
    # voting
    #############
    # dirs = [
    #     "Llama2-70b-ape-gsm8k-made-705673-2023-09-25-08:31/vote0",
    #     "Llama2-70b-ape-gsm8k-made-705673-2023-09-25-08:31/vote1",
    #     "Llama2-70b-ape-gsm8k-made-705673-2023-09-25-08:31/vote2",
    #     "Llama2-70b-ape-gsm8k-made-705673-2023-09-25-08:31/vote3",
    #     "Llama2-70b-ape-gsm8k-made-705673-2023-09-25-08:31/vote4",
    #     "Llama2-70b-ape-gsm8k-made-705673-2023-09-25-08:31/vote5",
    #     "Llama2-70b-ape-gsm8k-made-705673-2023-09-25-08:31/vote6",
    #     "Llama2-70b-ape-gsm8k-made-705673-2023-09-25-08:31/vote7",
    #     "Llama2-70b-ape-gsm8k-made-705673-2023-09-25-08:31/vote8",
    #     "Llama2-70b-ape-gsm8k-made-705673-2023-09-25-08:31/vote9",
    # ]
    # k = 10
    # # compute_accuracy("/mnt/cache/luzimu/code_generation-master/outs/Llama-2-7b-hf-math-2023-08-27-09:58/MATH_test_result.jsonl", "/mnt/cache/luzimu/code_generation-master/outs/Llama-2-7b-hf-math-2023-08-27-09:58/MATH_results", "/mnt/cache/luzimu/gsm8k-rft-llama7b-u13b_evaluation/MATH_test_orig.jsonl")
    # # compute_accuracy_k12("/mnt/cache/luzimu/code_generation-master/outs/Llama-2-7b-hf-k12-2023-08-28-17:38/4200_test/k12_test_result.jsonl", "/mnt/cache/luzimu/code_generation-master/outs/Llama-2-7b-hf-k12-2023-08-28-17:38/4200_test/")
    # for name in ["APE500", "GSM8K200"]:
    #     print(name + ":")
    #     in_files = [f"/mnt/cache/luzimu/code_generation-master/data/votings/{dir}/{name}/{name}_test_result.jsonl" for dir in dirs]
    #     out_files = [f"/mnt/cache/luzimu/code_generation-master/data/votings/{dir}/{name}/{name}_test_result.jsonl" for dir in dirs]
    #     for in_file, out_file in zip(in_files, out_files):
    #         compute_accuracy_ours(in_file, out_file, 5000)
