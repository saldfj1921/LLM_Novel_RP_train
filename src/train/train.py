from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, HfArgumentParser, Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset, concatenate_datasets
import os
import torch
from dataclasses import dataclass, field
import deepspeed
deepspeed.ops.op_builder.CPUAdamBuilder().load()
from torch.utils.tensorboard import SummaryWriter

# 确保环境变量中没有 WandB 的设置
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "dryrun"
    
# 自定义 Trainer 类以覆盖 compute_loss 方法
class CustomTrainer(Trainer):
    def __init__(self, *args, writer=None, deepspeed_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.writer = writer
        self.global_step = 0
        self.deepspeed_config = deepspeed_config
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Overwrite compute_loss method to log to TensorBoard
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs['labels']
        #channels = inputs['channels'][:, 0]
        channels = inputs['channels']

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
    
    def deepspeed_init(self, model):
        # 使用deepspeed初始化模型
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=self.deepspeed_config
        )
        return model_engine

class DataCollatorWithChannel:
    def __init__(self, tokenizer, mlm=False, mlm_probability=0.15):
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)

    def __call__(self, features):
        channels = [feature.pop('channels') for feature in features]
        batch = self.data_collator(features)
        batch['channels'] = torch.tensor([channels])
        return batch

# 调整数据集格式并添加 channel 信息
def preprocess_data(dataset, channel_label):
    MAX_LENGTH = 2048

    def instance_process_func(examples):
        tokenized_examples = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH)['input_ids']
        tokenized_examples = [ids[:MAX_LENGTH] for ids in tokenized_examples]  # 做一个截断
        channels = [channel_label] * len(tokenized_examples)
        return {'input_ids': tokenized_examples, 'labels': tokenized_examples, 'channels': channels}

    dataset = dataset.map(instance_process_func, batched=True, num_proc=4, remove_columns=['text'])
    return dataset

if "__main__" == __name__:
    model_path = "/data/jeriffli/LLM_weight/Qwen1.5-7B"

    # 使用DeepSpeed插件创建TrainingArguments
    training_args = TrainingArguments(
        output_dir="./results",
        logging_dir="./logs",
        logging_steps=10,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,  # 新增
        learning_rate=5e-05,
        weight_decay=0.01,
        bf16=True,
        remove_unused_columns=False,
        report_to=None,  # 禁用 W&B
        deepspeed="./ds_config.json"
    )

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    # 加载不同channel的数据
    channel_A_data = load_dataset('json', data_files='test_data/_1.json')
    channel_B_data = load_dataset('json', data_files='test_data/_2.json')

    train_dataset_A = preprocess_data(channel_A_data['train'], 0)
    train_dataset_B = preprocess_data(channel_B_data['train'], 1)

    # 合并数据集
    train_ds = concatenate_datasets([train_dataset_A, train_dataset_B])

    # 创建模型并以半精度形式加载
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.half, device_map={"": int(os.environ.get("LOCAL_RANK") or 0)})
    
    # 数据collator，用于动态padding
    data_collator = DataCollatorWithChannel(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 使用TensorBoard
    writer = SummaryWriter(log_dir='./logs')
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
    trainer.save_model(output_dir='./trained_llama')
    tokenizer.save_pretrained('./trained_llama')