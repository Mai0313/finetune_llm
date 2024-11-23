from typing import Any

from datasets import load_dataset
from src.config import config
from transformers import (
    Trainer,
    AutoTokenizer,
    TrainingArguments,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)

model_name = "meta-llama/Llama-3.2-1B-Instruct"

# 加载数据集
dataset = load_dataset(path="hugfaceguy0001/retarded_bar", name="question", cache_dir="./data")

# 初始化分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_name, cache_dir="./models", token=config.huggingface_token
)

# 添加特殊标记
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# 初始化模型
model = AutoModelForCausalLM.from_pretrained(
    model_name, cache_dir="./models", token=config.huggingface_token
)

# **调整模型的嵌入层大小**
model.resize_token_embeddings(len(tokenizer))


# 定义分词函数
def tokenize_function(examples: dict[str, Any]) -> dict[str, Any]:
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)


# 拆分数据集为训练集和验证集
split_datasets = dataset["train"].train_test_split(test_size=0.1, shuffle=True, seed=42)

# 对训练集进行分词
tokenized_train_dataset = split_datasets["train"].map(
    tokenize_function, batched=True, remove_columns=dataset["train"].column_names
)

# 对验证集进行分词
tokenized_eval_dataset = split_datasets["test"].map(
    tokenize_function, batched=True, remove_columns=dataset["train"].column_names
)

# 数据整理器
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=5000,
    save_total_limit=2,
    logging_steps=500,
    eval_strategy="steps",
    eval_steps=5000,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=1000,
    gradient_accumulation_steps=1,
    fp16=True,  # 如果硬件支持，使用混合精度训练
)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
)

# 开始训练
trainer.train()
