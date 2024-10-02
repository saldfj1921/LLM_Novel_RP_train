import os
import json

root_dir = '../pt_data_process'
pretrain_data_info = {}

# 遍历根目录下的所有子文件夹
for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)
    if os.path.isdir(subdir_path):
        print(f"Processing {subdir_path}...")
        output_dir = os.path.join(subdir_path, 'output')
        if os.path.exists(output_dir):
            pretrain_data_info[subdir] = {
                'token_num': 0,
                'content': []
            }
            # 查找 output 目录下的所有 JSON 文件
            for file_name in os.listdir(output_dir):
                if file_name.endswith('.json'):
                    file_path = os.path.join(output_dir, file_name)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            print(f"Processing {file_name}...")
                            data = json.load(file)
                            # 计算每个 JSON 文件中所有 text 字段的字符串总长度
                            token_num = sum(len(item['text']) for item in data if 'text' in item)
                            pretrain_data_info[subdir]['token_num'] += token_num
                            pretrain_data_info[subdir]['content'].append({
                                'path': file_path,
                                'token_num': token_num
                            })
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

# 打印结果
print(json.dumps(pretrain_data_info, ensure_ascii=False, indent=4))