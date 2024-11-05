import copy
import os
import json
from collections import defaultdict
import random
"""
(1) func for calc_status
"""
def update_category_info(category_info, file_path, instance_list):
    instance_num = len(instance_list)
    token_num = sum(len(instance['text']) for instance in instance_list)
    max_len = max(len(instance['text']) for instance in instance_list)

    category_info['file_paths'].append(file_path)
    category_info['instance_num'] += instance_num
    category_info['token_num'] += token_num
    category_info['max_len'] = max(category_info['max_len'], max_len)
    
def calculate_ratios(category_info, total_token_num, total_instances):
    category_info['token_ratio'] = str(round((category_info['token_num'] / total_token_num if total_token_num > 0 else 0)*100, 2))+'%'
    category_info['instance_ratio'] = str(round((category_info['instance_num'] / total_instances if total_instances > 0 else 0)*100, 2))+'%'
    for sub_category in category_info['sub_class'].values():
        calculate_ratios(sub_category, total_token_num, total_instances)

data_info_proto = {
        "file_paths": [],
        "instance_num": 0,
        "token_num": 0,
        "max_len": 0,
        "token_ratio": 0,
        "instance_ratio": 0,
        "sample_weight_rel": 1.0,
        "sub_class": {}}

def process_directory(processed_data_dir):
    data_stat = copy.deepcopy(data_info_proto)
    
    for root, _, files in os.walk(processed_data_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    instance_list = json.load(f)
                    categories = file.replace('.json', '').split('@')
                    
                    # 更新总信息
                    update_category_info(data_stat, file_path, instance_list)
                    
                    cur_node = data_stat
                    for category in categories[:-1]:
                        if category not in cur_node['sub_class'].keys():
                            cur_node['sub_class'][category] = deepcopy(data_info_proto)
                        update_category_info(cur_node, file_path, instance_list)
                        cur_node = cur_node['sub_class'][category]
                    update_category_info(cur_node, file_path, instance_list)
    
    total_token_num = data_stat['token_num']
    total_instances = data_stat['instance_num']
    
    # 计算百分比
    calculate_ratios(data_stat, total_token_num, total_instances)

    return data_stat

def remove_keys(node, keep_keys = []):
    for k in data_info_proto.keys():
        if k != 'sub_class' and not k in keep_keys and k in node:
            node.pop(k)
    for child in node['sub_class'].values():
        remove_keys(child, keep_keys)

def calc_data_stat(processed_data_dir, data_stat_path):
    """
      get data distribution info
    """
    data_stat = process_directory(processed_data_dir)
    remove_keys(data_stat, keep_keys = ['token_ratio', 'token_num', 'sample_weight_rel'])

    with open(data_stat_path, 'w', encoding='utf-8') as file:
        json.dump(data_stat, file, indent=4, ensure_ascii=False)

"""
(2) func for calc sample config
"""
def calc_sample_ratio(node):
    sum_weight = 0
    for child in node['sub_class'].values():
        sum_weight += child['sample_weight_rel']
    for child in node['sub_class'].values():
        child['sample_target_ratio'] = node['sample_target_ratio'] * (child['sample_weight_rel']/sum_weight)       # 占总体的比例
        child['require_token_num'] = node['require_token_num'] * (child['sample_weight_rel']/sum_weight)                  # 为达到这个比例需要的token数
        child['sample_keep_rate'] = float(child['require_token_num'])/child['token_num']                           # 为达到token数的采样比例
    for child in node['sub_class'].values():
        calc_sample_ratio(child)

def calc_sample_config(data_stat_path, sample_config_path, total_require_token_num):
    with open(data_stat_path, 'r', encoding='utf-8') as f:
        sample_config = json.load(f)
    sample_config['sample_target_ratio'] = 1.0
    sample_config['require_token_num'] = total_require_token_num
    calc_sample_ratio(sample_config)
    
    with open(sample_config_path, 'w', encoding='utf-8') as file:
        json.dump(sample_config, file, indent=4, ensure_ascii=False)

"""
(3) mix data depend on sample_config
"""
class DataCollector:
    def __init__(self, output_dir, dataset_name='train_data', buff_size = 2024000, max_instance_num=1024):
        self.buf = []
        self.max_instance_num = max_instance_num
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.split_cnt = 1
        self.buff_size = buff_size
        self.stat = {}

    def savefile(self, instances):
        for instance in instances:
            if instance['channel_name'] not in self.stat:
                self.stat[instance['channel_name']] = {'instance_num':0, 'token_num':0}    
            self.stat[instance['channel_name']]['instance_num'] += 1
            self.stat[instance['channel_name']]['token_num'] += len(instance['text'])
        with open(self.output_dir+self.dataset_name+'_'+str(self.split_cnt)+'.json', 'w', encoding='utf-8') as file:
            print('save', self.output_dir+self.dataset_name+'_'+str(self.split_cnt), '\ndata stat;\n', self.stat)
            json.dump(instances, file, indent=4, ensure_ascii=False)
            self.split_cnt += 1
    
    def dump(self, dump_all=False):
        random.shuffle(self.buf)
        while len(self.buf) > self.max_instance_num:
            self.savefile(self.buf[:self.max_instance_num])
            self.buf = copy.deepcopy(self.buf[self.max_instance_num:])
        if dump_all:
            self.savefile(self.buf)

    def add(self, instance):
        self.buf.append(instance)
        if len(self.buf)%1000==0: print(len(self.buf))
        if len(self.buf) > self.buff_size:
            self.dump()  

def mix_dataset(processed_data_dir, sample_config_path, channel_info_path, output_dir, total_require_token_mul=1.0):
    channel_d = {}
    channel_cnt = 0
    with open(sample_config_path, 'r', encoding='utf-8') as f:
        sample_config = json.load(f)
    dc = DataCollector(output_dir, max_instance_num=102400)
        
    for root, _, files in os.walk(processed_data_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    print(file_path)
                    instance_list = json.load(f)
                    categories = file.replace('.json', '').split('@')
                    
                    cur_node = sample_config
                    channel_name = categories[0] # use first class as channel
                    if channel_name not in channel_d:
                        channel_d[channel_name] = channel_cnt
                        channel_cnt += 1
                    channel_id = channel_d[channel_name]

                    for category in categories[:-1]:
                        cur_node = cur_node['sub_class'][category]
                    sample_keep_rate = cur_node['sample_keep_rate']
                    for instance in instance_list:
                        #print(random.random(), sample_keep_rate)
                        if random.random() < sample_keep_rate * total_require_token_mul:
                            # add tag
                            instance['channel_name'] = channel_name
                            instance['channel'] = channel_id
                            dc.add(instance)            
        dc.dump(dump_all=True)
        
        with open(channel_info_path, 'w', encoding='utf-8') as file:
            json.dump(channel_d, file, indent=4, ensure_ascii=False)



if __name__ == "__main__":
    processed_data_dir = "../../data/pt_processed_data/"
    data_stat_path = "./data_stat.json"
    sample_config_path = "./sample_config.json"
    channel_info_path = "./channel_info.json"
    output_dir = "../../data/pt_train_data/"

    #calc_data_stat(processed_data_dir, data_stat_path)

    calc_sample_config(data_stat_path, sample_config_path, total_require_token_num=3000065723)

    mix_dataset(processed_data_dir, sample_config_path, channel_info_path, output_dir)
