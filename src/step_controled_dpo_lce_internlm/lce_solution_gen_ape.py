import re
import json
import os
import sys
from io import StringIO
import threading

from tqdm import tqdm
from multiprocessing import Pool,RLock
from huggingface_hub import InferenceClient
from jupyter_client.manager import start_new_kernel
import zmq
import time
from argparse import ArgumentParser
import requests

import copy

def timestamp() -> str:
    nowtime = time.strftime('-%Y%m%d-%H%M', time.localtime(time.time()))
    print(nowtime)  
    return nowtime  

def save_jsonl(data: list, path: str, mode='w', add_timestamp=True, verbose=True) -> None:
    if add_timestamp:
        file_name = f"{path.replace('.jsonl','')}{timestamp()}.jsonl"
    else:
        file_name = path
    with open(file_name, mode, encoding='utf-8') as f:
        if verbose:
            for line in tqdm(data, desc='save'):
                f.write(json.dumps(line, ensure_ascii=False) + '\n')
        else:
            for line in data:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')


def load_jsonl(path: str):
    with open(path, "r", encoding='utf-8') as fh:
        return [json.loads(line) for line in fh.readlines() if line]
    

class JupyterNotebookKernel(object):

    lock = RLock()

    def __init__(self, retries=5, delay=5):
        JupyterNotebookKernel.lock.acquire()
        for _ in range(retries):
            try:
                self.manager, self.client = start_new_kernel(kernel_name='python')
                break
            except zmq.ZMQError as e:
                if "Address already in use" in str(e) and _ < retries - 1:  # check if the error is because the address is in use
                    print(f"Address already in use. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    raise
        else:
            raise Exception("Failed to start kernel after multiple retries.")
        JupyterNotebookKernel.lock.release()
                

    
    def shutdown(self):
        if self.manager:
            self.manager.shutdown_kernel()
            self.manager = None
            self.client = None


    def handle_iopub_msg(self):
        result = ''

        while msg := self.client.get_iopub_msg(timeout=10):
            
            if msg['msg_type'] == 'status' and msg['content']['execution_state'] == 'idle':
                break

            if msg['msg_type'] == 'stream':
                result += msg['content']['text']
            
            if msg['msg_type'] == 'execute_result':
                result += msg['content']['data']['text/plain']
            
            if msg['msg_type'] == 'error':
                if isinstance(msg['content']['traceback'], list):
                    msg['content']['traceback'] = ' '.join(msg['content']['traceback'])

                error = re.sub(
                    '\x1B\\[([0-9]{1,2}(;[0-9]{1,2})?)?[mGK]',
                    '',
                    msg['content']['traceback'],
                )

                result += error
        
        if len(result) == 0:
            result = '<empty_execution>'

        return result.strip()

    def run_code(self, code):
        try:
            self.client.execute(code, allow_stdin=False, reply=True, timeout=6)
            return self.handle_iopub_msg()
        except zmq.ZMQError as e:
            if "Address already in use" in str(e):
                print("Address already in use. Restarting kernel...")
                self.shutdown()
                self.__init__()
                return self.run_code(code)
            else:
                raise
        except Exception as e:
            return f'{"-"*75} {str(e)}{" "*32}Traceback (most recent call last) '

    def monitor_errors(self):
        old_stderr = sys.stderr
        sys.stderr = captured_stderr = StringIO()
        while True:
            # Check the error stream every second (adjust as needed)
            time.sleep(1)
            error_output = captured_stderr.getvalue()
            if "[IPKernelApp] WARNING | Parent appears to have exited, shutting down." in error_output:
                # Do your restart logic here
                os.execl(sys.executable, sys.executable, *sys.argv)

    def start_monitoring(self):
        # This starts the error monitor in a separate thread
        error_monitor_thread = threading.Thread(target=self.monitor_errors)
        error_monitor_thread.daemon = True  # So the thread will exit when the main program exits
        error_monitor_thread.start()


class API:

    def __init__(self, ip = '', port='8001'):
        self.api_url = f"http://{ip}:{port}/generate"
        self.headers = {"User-Agent": "Test Client"}

    def get_result(self, inputs, parameters=None):
        
        local_parameters = dict(prompt=inputs, max_tokens=1024, stream=False)

        if parameters is not None:
            local_parameters.update(parameters)
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=local_parameters, stream=False)
            data = json.loads(response.content)
            text = data["text"][0]

            if text.startswith(inputs):
                text = text[len(inputs):]

            return text
        
        except:
            import traceback
            traceback.print_exc()
            print(inputs) 
            return None
        
def get_prompt(messages):
    prompt = f'<|system|><|text|><|endofblock|><|endofmessage|><|user|><|text|>{messages[1]["content"]}<|endofblock|><|endofmessage|><|assistant|>'
    for block in messages[2:]:
        if block["role"] == "text":
            prompt += f'<|text|>{block["content"]}<|endofblock|>'
        elif block["role"] == "code":
            prompt += f'<|code|>{block["content"]}<|endofblock|>'
        elif block["role"] == "execution":
            prompt += f'<|execution|>{block["content"]}<|endofblock|>'
            
    return prompt


def code_generation(messages_to_be_run):
    messages = copy.deepcopy(messages_to_be_run)
    prompt = get_prompt(messages)

    jupyter = JupyterNotebookKernel()
    jupyter.start_monitoring()

    for block in messages_to_be_run:
        if block["role"] == "code":
            execution = jupyter.run_code(block["content"])
            # print(f"EXECUTION: {execution}")

    code = ''
    for i in range(64):

        temperature = min(1.0 + i * 0.05, 1.2)

        parameters=dict(
            use_beam_search=False,
            n=1,
            temperature=temperature,
            stop=['<|endofmessage|>', '<|endofblock|>'], 
            include_stop_str_in_output=True,
            skip_special_tokens=False,
        )
        
        result = api.get_result(prompt, parameters=parameters)
        prompt += result

        if result is None:
            messages.append({'role': 'exceed_max_length/return_first_code', 'content': code})
            jupyter.shutdown()
            return messages

         
        results = result.split('<|')
        for id, sub_result in enumerate(results):
            if id > 0:
                sub_result = '<|' + sub_result
            if len(sub_result.replace('<|assistant|>','').replace('<|code|>','').replace('<|text|>','').replace('<|endofblock|>','')) == 0:
                continue
            if sub_result.startswith('<|code|>'):
                code = sub_result.replace('<|code|>', '').replace('<|endofblock|>', '')
                messages.append({'role': 'code', 'content': code})
                
                execution = jupyter.run_code(code)

                prompt += f"<|execution|>{execution}<|endofblock|>"
                messages.append({'role': 'execution', 'content': execution})
            elif not sub_result.endswith('<|endofmessage|>'):
                messages.append({'role': 'text', 'content': sub_result.replace('<|text|>', '').replace('<|endofblock|>', '')})
        if  result.endswith('<|endofmessage|>') or result.endswith('<|endoftext|>'):
            break
    
    jupyter.shutdown()
    return messages

def process_full(data, key="messages_to_be_run"):
    # if no correct solution, try to generate one
    
    query = data[key]

    debug_result = code_generation(query)

    data['debug_result'] = debug_result

    return data


if __name__ == '__main__':
    parser = ArgumentParser(description="A simple argument parser")
    parser.add_argument("r", type=int, help="round number")
    parser.add_argument("-i", type=str, help="index within a round")
    parser.add_argument("-a", type=str, help="ip address")

    args = parser.parse_args()

    ip = args.a

    print(ip)
    
    api = API(port="8001", ip=ip)

    input_path = f'data/lce_solutions/internlm_negative_positive_gen/sc_dpo/ape/to_be_run_{args.i}_round{args.r}.jsonl'
    output_path = f'data/lce_solutions/internlm_negative_positive_gen/sc_dpo/ape/{args.i}_round{args.r}.jsonl'

    # output_path = f'/mnt/cache/wangke/code_generation/outs/debug/{name}/{name}_test_result.jsonl'
    if not os.path.exists("/".join(output_path.split("/")[:-1])):
        os.makedirs("/".join(output_path.split("/")[:-1]))
    
    try:
        all = load_jsonl(output_path)
    except FileNotFoundError:
        all = []

    BEGIN = len(all)

    OVER_WRITE = True
    humaneval = load_jsonl(input_path)
    END = len(humaneval)
    outs = []

    # process_full(humaneval[0])

    counter = BEGIN
    while counter < END:
        pool = Pool(8)
        try:
            results = pool.imap(process_full, humaneval[BEGIN:END])
            for d in tqdm(results, total=len(humaneval[BEGIN:END])):
                outs.append(d)
                all.append(d)
                counter += 1
                if counter % 10 == 0 or counter == END:
                    if counter <= 10 and OVER_WRITE:
                        save_jsonl(outs, output_path,mode='w', add_timestamp=False, verbose=False)
                    else:
                        save_jsonl(outs, output_path,mode='a', add_timestamp=False, verbose=False)
                    outs = []
                    BEGIN = counter
        except Exception as e:
            print(f'<|{str(e)}|>')
            pool.terminate()  # 立即终止所有子进程
            print(f"[restarting]")
            os.execl(sys.executable, sys.executable, *sys.argv)

        finally:
            pool.close()  # 关闭pool，防止新的任务提交到pool
            pool.join()   # 等待子进程结束

    
    print('Total: ', counter)
