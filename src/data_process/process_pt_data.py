import glob
import os
import json
import random
import re
import pandas as pd

BaseN = 20000000

DEFAULT_CONTENT_KEY = 'text'

DEFAULT_TYPE_KEY = 'default'
FILENAME_TYPE_KEY = '@filename'

JSONL_FORMAT = 'jsonl'
PARQUET_FORMAT = 'parquet'
TXT_FORMAT = 'txt'

def is_novel_line_filter(line):
    if line.strip() == "": return True
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    if url_pattern.search(line):
        return True

PT_DATASET_DIR = '../../data/pt_raw_data/'
OUTPUT_DIR = '../../data/pt_processed_data/'

class DataCollector:
    def __init__(self, dataset_name, maxsize_d, drop_ratio=0, content_keys=[], type_key=None, head_tag='g'):
        self.dataset_name = dataset_name
        # hyper-parameters
        self.MaxSizeOneFile = 10 * BaseN
        self.MinSizeOneFile = 1000000
        self.drop_ratio = drop_ratio
        self.head_tag = head_tag
        # set dataset info
        self.content_keys = content_keys
        self.type_key = type_key
        # init buff and status
        self.data_buff_d = {}
        self.ins_cnt_d = {}
        self.size_d = {}
        for dtype, maxsize in maxsize_d.items():
            self.data_buff_d[dtype] = []
            self.ins_cnt_d[dtype] = maxsize
            self.size_d[dtype] = 0
        self.file_split_id = 0
    
    def set_cur_filepath(self, file_path):
        self.file_path = file_path

    def _get_filename_key(self,):
        dtype = None
        for candicate_dtype in self.data_buff_d.keys():
            if candicate_dtype in self.file_path:
                dtype = candicate_dtype
                break
        if dtype == None: raise ValueError(f'{self.file_path} not in {self.data_buff_d.keys()}')
        return dtype
    
    def add(self, content_d):
        """
        content_d: dict of one instance, include content and type(optional)
        """
        if random.random() < self.drop_ratio: return
        
        # analysis content & type
        content_text = '\t'.join([content_d[k] for k in self.content_keys])
        if self.type_key == None:
            dtype = DEFAULT_TYPE_KEY
        elif self.type_key == FILENAME_TYPE_KEY:
            dtype = self._get_filename_key()
        else:
            dtype = self.content_d[self.type_key]
        
        # finish dtype, skip
        if self.ins_cnt_d[dtype] < 0: return

        # add to buff
        self.data_buff_d[dtype].append({'text': content_text})
        self.size_d[dtype] += len(content_text)
        if self.size_d[dtype] > self.MaxSizeOneFile:
            self.dump(dtype)
    
    def dump(self, dtype):
        if (self.size_d[dtype] < self.MinSizeOneFile):
            print(f'dtype {dtype} data_buff_d size {self.size_d[dtype]} < MinSizeOneFile {self.MinSizeOneFile}, not dump')
            return
        print('dump', dtype, 'size', self.size_d[dtype], 'ins num', len(self.data_buff_d[dtype]), 'rest', self.ins_cnt_d)
        self.file_split_id += 1
        o_fname = self.head_tag+'@'+self.dataset_name+'@'+dtype+'@'+'_'+str(self.file_split_id)+'.json'
        with open(OUTPUT_DIR+o_fname, 'w', encoding='utf-8') as file:
            #json.dump(self.data_buff_d[dtype], file, ensure_ascii=False)
            json.dump(self.data_buff_d[dtype], file, ensure_ascii=False, indent=4)
        self.data_buff_d[dtype] = []
        self.ins_cnt_d[dtype] -= self.size_d[dtype]
        self.size_d[dtype] = 0
    
    def dump_all(self,):
        for dtype in self.data_buff_d.keys():
            self.dump(dtype)
    
    def is_file_finish(self,):
        # if type depend on filename, once cur type finish , then skip file 
        if self.type_key == FILENAME_TYPE_KEY: 
            dtype = self._get_filename_key()
            if self.ins_cnt_d[dtype] < 0: 
                print(f'@type {dtype} reach maximum ! skip file, {self.file_path}')
                return True
        return self.is_finish()
    
    def is_finish(self,):
        # when all type ins_cnt < 0, finish
        if sum([v for v in self.ins_cnt_d.values()]) < 0:
            print('@all type reach maximum ! finish !')
            return True
        return False

    def show(self,): 
        print('rest', self.ins_cnt_d)
        print('cur data buff')
        for dtype in self.data_buff_d.keys():
            print('dtype', dtype, ': size', self.size_d[dtype], 'ins num', len(self.data_buff_d[dtype]))
        


def process(dataset_name, format, maxsize_d, content_keys = [], type_key=None, drop_ratio=0.0, head_tag='g'):
    """
    dataset_name: dataset name, must be the same with filename
    format: support jsonl, parquet, txt
    maxsize_d: max total token num for each sub_class
    content_key: the name of the col name contain main context
    type_key:  the name of the col contain type (if sub_class exist)
    drop_ratio: random dropout
    head_tag: first class name
    """
    # setting for different format
    if format == TXT_FORMAT:
        content_keys = [DEFAULT_CONTENT_KEY]
    else:
        assert len(content_keys) > 0
    # init data collector
    dc = DataCollector(dataset_name, maxsize_d, drop_ratio=drop_ratio, content_keys=content_keys, type_key=type_key, head_tag=head_tag)
    dataset_path = os.path.join(PT_DATASET_DIR, dataset_name)

    print('process', dataset_name, '...')
    file_paths = glob.glob(os.path.join(dataset_path, '**', '*.'+format), recursive=True)
    # main process
    for file_path in file_paths:   
        dc.set_cur_filepath(file_path)
        dc.show()
        if dc.is_file_finish(): continue

        print('process file', file_path)
        if format == JSONL_FORMAT:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    content_d = json.loads(line)
                    dc.add(content_d)
                    if dc.is_file_finish(): break
        if format == PARQUET_FORMAT:
            df = pd.read_parquet(file_path)
            print("Columns:", df.columns.tolist())
            for index, row in df.iterrows():
                content_d = row.to_dict()
                dc.add(content_d)
                if dc.is_file_finish(): break
        if format == TXT_FORMAT:   
            try_encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030']
            for encoding in try_encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        # notice load txt is different with other format, we concate all lines together and then process
                        # because we suppose all txt is small 
                        # and there is not is_file_finish break
                        lines = []
                        for line in file:
                            if not is_novel_line_filter(line):
                                lines.append(line)
                        lines = [line.strip() for line in lines if line.strip() != ""]
                        # chunk
                        NovelChunkSize = 100*1000
                        text = ''
                        for line in lines:
                            text += line + '\n'
                            if len(text) > NovelChunkSize:
                                dc.add({DEFAULT_CONTENT_KEY: text})
                                text = ''
                        dc.add({DEFAULT_CONTENT_KEY: text})
                        break
                except:
                    continue
        if dc.is_finish(): break
    dc.dump_all()

#  ============ CN General ===================
def process_Skypile():
    maxsize_d = {DEFAULT_TYPE_KEY: 100*BaseN}
    process('Skypile', JSONL_FORMAT, maxsize_d, content_keys = ['text'], head_tag='CN_general')

def process_Tiger():
    #Tiger有分类
    maxsize_d = {
        'zh-news': 20*BaseN,
        'zh-books': 20*BaseN,
        'zh-baike': 20*BaseN,
        'zh-zhihu': 20*BaseN,
        'zh-webtext': 20*BaseN,
        'en-book': 5*BaseN,
        'en-webtext': 5*BaseN,
        'en-wiki': 5*BaseN,
        'en-github': 5*BaseN,
        'en-stackoverflow': 5*BaseN
    }
    process('Tiger', PARQUET_FORMAT, maxsize_d, content_keys = ['title', 'content'], type_key = 'dataType', head_tag='CN_general')

#  ============ ENG General ===================
def process_WanJuan2():
    maxsize_d = {DEFAULT_TYPE_KEY: 500*BaseN}
    process('WanJuan2', JSONL_FORMAT, maxsize_d, content_keys = ['content'], head_tag='ENG_general')

#  ============ Code ===================
def process_starcode():
    maxsize_d = {'cpp': 120*BaseN,
                'java': 120*BaseN,
                'python': 120*BaseN,}
    process('startcoder', PARQUET_FORMAT, maxsize_d, content_keys = ['content'], type_key = FILENAME_TYPE_KEY, head_tag='code')

#  ============ Math ===================
def process_Dolma_math():
    maxsize_d = {DEFAULT_TYPE_KEY: 400*BaseN}
    process('Dolma-math', JSONL_FORMAT, maxsize_d, content_keys = ['text'], drop_ratio=0.5, head_tag='math')

#  ============ CN spec ===================
def process_MNBVC_novel():
    maxsize_d = {'名著':  20*BaseN,
                '杂书3': 10*BaseN,
                '杂书4': 10*BaseN,
                '杂书3': 10*BaseN, 
                '杂书5': 10*BaseN, 
                '杂书6': 10*BaseN, 
                '知名网络小说': 30*BaseN, 
                '科幻小说': 20*BaseN,
                '网络小说2': 30*BaseN, 
                '轻小说': 30*BaseN}
    process('MNBVC_novel', TXT_FORMAT, maxsize_d, type_key = FILENAME_TYPE_KEY, head_tag='novel')


#  ============ GN ===================
def process_My_novel():
    maxsize_d = {DEFAULT_TYPE_KEY:  50*BaseN}
    process('My_novel', TXT_FORMAT, maxsize_d, head_tag='My')

#process_Skypile()
#process_Tiger()
#process_WanJuan2()
#process_starcode()
#process_Dolma_math()
process_MNBVC_novel()
#process_My_novel()