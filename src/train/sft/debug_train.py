import glob
from time import sleep
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, HfArgumentParser, Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from peft import LoraConfig,get_peft_model
from datasets import load_dataset, concatenate_datasets
import os
import torch
from dataclasses import dataclass, field
from torch.utils.tensorboard import SummaryWriter
from config import *

# 确保环境变量中没有 WandB 的设置
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "dryrun"
def instance_process_func(example):
    """
    for Qwen template
    """
    MAX_LENGTH = 2048
    IM_END="<|im_end|>"
    IM_START="<|im_start|>"
    TURN_SPLIT="\n"
    ROLE_TOKEN = {"system": "system\n",
                "user": "user\n",
                "assistant": "assistant\n"}

    input_ids, attention_mask, labels = [], [], []
    input_str = ""
    #print('==========')
    #print(example['text'])
    # process multi-turn template
    for tid, term in enumerate(example['text']):
        #print('@', term)
        if tid == 0:
            prefix_tokens = tokenizer(IM_START+ROLE_TOKEN[term['role']], add_special_tokens=False)
            input_str += IM_START+ROLE_TOKEN[term['role']] + term['content'] + IM_END
        else:
            prefix_tokens = tokenizer(TURN_SPLIT+IM_START+ROLE_TOKEN[term['role']], add_special_tokens=False)
            input_str += TURN_SPLIT+IM_START+ROLE_TOKEN[term['role']] + term['content'] + IM_END
        content_tokens = tokenizer(term['content']+IM_END, add_special_tokens=False)        # content 就是要预测的内容，注意IM_END也是要预测的

        input_ids += prefix_tokens['input_ids']+content_tokens['input_ids']
        attention_mask += prefix_tokens['attention_mask']+content_tokens['attention_mask']

        labels += [-100] * len(prefix_tokens['input_ids'])
        if term['role'] == 'assistant':
            labels +=  content_tokens['input_ids']
        else:
            labels += [-100] * len(content_tokens['input_ids'])

    #print(f'input_ids:{input_ids}\nlabels:{labels}\nattention_mask:{attention_mask}\n')
    #print(input_str)

    # channel
    channel = example['channel']

    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "channels": channel,
    }

def preprocess_data(dataset):
    # SFT处理规则比较多，这里一条条处理（batched=False），对性能影响待评估
    dataset = dataset.map(instance_process_func, batched=False, num_proc=4, remove_columns=['text', 'channel_name'])
    return dataset

# 加载数据，并分channel
def load(data_dir):
    dataset_list = []
    file_paths = glob.glob(os.path.join(data_dir, '*.json'), recursive=True)
    for path in file_paths:
        single_dataset = load_dataset('json', data_files=path)
        single_pc_dataset = preprocess_data(single_dataset['train'])
        dataset_list.append(single_pc_dataset)

    # 合并数据集
    train_ds = concatenate_datasets(dataset_list)
    return train_ds

class DataCollatorWithChannel:
    def __init__(self, tokenizer, mlm=False, mlm_probability=0.15):
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    def __call__(self, features):
        channels = [feature.pop('channels') for feature in features]
        batch = self.data_collator(features)
        batch['channels'] = torch.tensor(channels)
        return batch

# 自定义 Trainer 类以覆盖 compute_loss 方法
class CustomTrainer(Trainer):
    def __init__(self, *args, writer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.writer = writer
        self.global_step = 0
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        #print(inputs)
        channels = inputs.pop('channels')
        # Overwrite compute_loss method to log to TensorBoard
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs['labels']

        # Flatten logits and labels for calculating per-instance loss
        shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = labels[..., 1:].contiguous().view(-1)
        
        # Calculate per-token loss and then sum them up for each instance
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        per_token_loss = loss_fct(shift_logits, shift_labels)
        per_instance_loss = per_token_loss.view(logits.size(0), logits.size(1) - 1).sum(dim=1).detach().cpu()

        # Record the loss for each channel
        channel_losses = {}
        for loss, channel in zip(per_instance_loss, channels):
            channel = channel.item()
            if channel not in channel_losses:
                channel_losses[channel] = []
            channel_losses[channel].append(loss.item())
        
        # Log the average loss for each channel
        for channel, losses in channel_losses.items():
            avg_loss = sum(losses) / len(losses)
            self.writer.add_scalar(f'Loss/Channel_{channel}', avg_loss, self.global_step)

        self.global_step += 1
        return (outputs.loss, outputs) if return_outputs else outputs.loss
                        
if "__main__" == __name__:
    # # ===== debug process_func =====
    # model_path = MODEL_PATH
    # tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    # test_example = {'text': [{"role": "system", "content": "请你扮演一个猫咪"},
    #                         {"role": "user", "content": "你好"},
    #                         {"role": "assistant", "content": "喵喵"},
    #                         {"role": "user", "content": "你是谁"},
    #                         {"role": "assistant", "content": "喵喵喵"}],
    #                 'channel': 0,
    #                 'channel_name': "LCCC"}
    # instance_process_func(test_example)

    #  ===== train =====
    model_path = MODEL_PATH
    DATA_DIR = "/mnt/data/jeriffli/farlight84llm/dataset/mydebug"
    USE_LORE = True

    training_args = TrainingArguments(
        report_to=None,  # 禁用 W&B
        remove_unused_columns=False,
        output_dir=OUTPUT_PATH,
        logging_dir=LOG_PATH,
        logging_steps=10,
        save_steps=500,
        save_total_limit=1,
        bf16=True,
        num_train_epochs=100,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,  # 新增
        gradient_checkpointing=True,
        learning_rate=5e-05,
        weight_decay=0.01
    )

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    # 加载不同channel的数据
    train_ds = load(DATA_DIR)

    # 创建模型并以半精度形式加载
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.half, attn_implementation='flash_attention_2', device_map={"": int(os.environ.get("LOCAL_RANK") or 0)})
    model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法
    if USE_LORE:
        lora_config = LoraConfig(
            r=32,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj","down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    # 数据collator，用于动态padding
    data_collator = DataCollatorWithChannel(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 使用TensorBoard
    writer = SummaryWriter(log_dir=LOG_PATH)
    # 使用trainer训练
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
        writer=writer,
        )
    trainer.train() # 开始训练
    # 训练结束后关闭TensorBoard记录器
    writer.close()

    # 保存训练后的模型和tokenizer
    trainer.save_model(output_dir=MODEL_OUTPUT_PATH)
    tokenizer.save_pretrained(MODEL_OUTPUT_PATH)