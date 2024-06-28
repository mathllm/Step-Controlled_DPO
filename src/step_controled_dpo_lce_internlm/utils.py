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