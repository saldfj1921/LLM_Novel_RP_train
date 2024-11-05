import glob
import os
import json
import random
import re
import pandas as pd

DEFAULT_CONTENT_KEY = 'text'

DEFAULT_TYPE_KEY = 'default'
FILENAME_TYPE_KEY = '@filename'

JSON_FORMAT = 'json'
JSONL_FORMAT = 'jsonl'
PARQUET_FORMAT = 'parquet'
TXT_FORMAT = 'txt'

SFT_DATASET_DIR = '../../data/sft_raw_data/'
OUTPUT_DIR = '../../data/sft_processed_data/'

def process(datase_name, process_func, format):
    sft_data = []
    file_paths = glob.glob(os.path.join(os.path.join(SFT_DATASET_DIR, datase_name), '**', '*.'+format), recursive=True)
    for file_path in file_paths:
        # TODO: sub_class mix
        print('process', file_path)
        if format == JSONL_FORMAT:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    instance = json.loads(line)
                    instance = process_func(instance)
                    sft_data.append(instance)
        elif format == JSON_FORMAT:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    instance_list = json.load(file)
            except:     # back to jsonl
                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        instance = json.loads(line)
                        instance = process_func(instance)
                        sft_data.append(instance)
            with open(file_path, 'r', encoding='utf-8') as file:
                instance_list = json.load(file)
                for instance in instance_list:
                    instance = process_func(instance)
                    sft_data.append(instance)
        elif format == PARQUET_FORMAT:
            df = pd.read_parquet(file_path)
            print("Columns:", df.columns.tolist())
            for index, row in df.iterrows():
                instance = row.to_dict()
                instance = process_func(instance)
                sft_data.append(instance)
    # dump
        with open(os.path.join(OUTPUT_DIR, datase_name)+'@_.json', 'w', encoding='utf-8') as file:
            #json.dump(self.data_buff_d[dtype], file, ensure_ascii=False)
            json.dump(sft_data, file, ensure_ascii=False, indent=4)
    return sft_data

S = 'system'
U = 'user'
A = 'assistant'
#pCLUE
def func_pclue(org_instance):
    instance = [{'role': S, 'content': ''}, 
                {'role': U, 'content': org_instance['input']},
                {'role': A, 'content': org_instance['target']}]
    return instance
#process('pCLUE', func_pclue, JSON_FORMAT)

#alpaca_GPT4
def func_alpaca_gpt4(org_instance):
    if 'input' in org_instance and len(org_instance['input']) > 0:
        instance = [{'role': S, 'content': org_instance['instruction']}, 
                    {'role': U, 'content': org_instance['input']},
                    {'role': A, 'content': org_instance['output']}]
    else:
        instance = [{'role': S, 'content': ''}, 
                    {'role': U, 'content': org_instance['instruction']},
                    {'role': A, 'content': org_instance['output']}]
    return instance
#process('alpaca_GPT4', func_alpaca_gpt4, JSON_FORMAT)

#COIG
def func_COIG(org_instance):
    if 'textbox_question' in org_instance:
        if 'textbox_q_context' not in org_instance or type(org_instance['textbox_q_context']) != str: org_instance['textbox_q_context']=''
        if 'textbox_answer_analysis' not in org_instance or type(org_instance['textbox_answer_analysis']) != str: org_instance['textbox_answer_analysis']=''
        instance = [{'role': S, 'content': org_instance['textbox_q_instruction']}, 
                    {'role': U, 'content': org_instance['textbox_q_context']+org_instance['textbox_question']},
                    {'role': A, 'content': org_instance['textbox_answer']+org_instance['textbox_answer_analysis']}]
        return instance
    
    if 'input' in org_instance and len(org_instance['input']) > 0:
        instance = [{'role': S, 'content': org_instance['instruction']}, 
                    {'role': U, 'content': org_instance['input']},
                    {'role': A, 'content': org_instance['output']}]
    else:
        instance = [{'role': S, 'content': ''}, 
                    {'role': U, 'content': org_instance['instruction']},
                    {'role': A, 'content': org_instance['output']}]
    return instance
#process('COIG', func_COIG, JSONL_FORMAT)

#COIG-CQIA
def func_COIG_CQIA(org_instance):
    if 'input' in org_instance and len(org_instance['input']) > 0:
        instance = [{'role': S, 'content': org_instance['instruction']}, 
                    {'role': U, 'content': org_instance['input']},
                    {'role': A, 'content': org_instance['output']}]
    else:
        instance = [{'role': S, 'content': ''}, 
                    {'role': U, 'content': org_instance['instruction']},
                    {'role': A, 'content': org_instance['output']}]
    return instance
#process('COIG-CQIA', func_COIG_CQIA, JSONL_FORMAT)

#firefly
def func_firefly(org_instance):
    instance = [{'role': S, 'content': ''}, 
                {'role': U, 'content': org_instance['input']},
                {'role': A, 'content': org_instance['target']}]
    return instance
#process('firefly', func_firefly, JSONL_FORMAT)

#FLAN
def func_FLAN(org_instance):
    instance = [{'role': S, 'content': ''}, 
                {'role': U, 'content': org_instance['inputs']},
                {'role': A, 'content': org_instance['targets']}]
    return instance
#process('FLAN', func_FLAN, JSONL_FORMAT)

#GPTeacher
def func_GPTeacher(org_instance):
    if 'response' in org_instance: 
        a_content = org_instance['response']
    if 'output' in org_instance: 
        a_content = org_instance['output']
    instance = [{'role': S, 'content': org_instance['instruction']}, 
                {'role': U, 'content': org_instance['input']},
                {'role': A, 'content': a_content}]
    return instance
#process('GPTeacher', func_GPTeacher, JSON_FORMAT)

#WizardLM
def func_WizardLM(org_instance):
    instance = [{'role': S, 'content': ''}, 
                {'role': U, 'content': org_instance['conversation'][0]['human']},
                {'role': A, 'content': org_instance['conversation'][0]['assistant']}]
    return instance
#process('WizardLM_evol_instruct', func_WizardLM, JSONL_FORMAT)

#LCCC
CHAT_SYSTEMS = ['请你模拟两个人进行对话', '请你生成一段聊天', '请你扮演一个人和另一个人聊天', '请你进行一段对话', '下面是两个人在聊天', '下面是一段对话', '两个人在网上聊天']
def func_LCCC(org_instance):
    instance = [{'role': S, 'content': random.choice(CHAT_SYSTEMS)}]
    for i, content in enumerate(org_instance):
        if i % 2 == 0:
            instance.append({'role': U, 'content': content})
        else:
            instance.append({'role': A, 'content': content})
    return instance
process('LCCC', func_LCCC, JSONL_FORMAT)