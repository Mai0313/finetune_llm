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
dataset = load_dataset(path="hugfaceguy0001/retarded_bar", name="question", cache_dir="./data")

tokenizer = AutoTokenizer.from_pretrained(
    model_name, cache_dir="./models", token=config.huggingface_token
)
model = AutoModelForCausalLM.from_pretrained(
    model_name, cache_dir="./models", token=config.huggingface_token
)


def tokenize_function(examples: dict[str, Any]) -> dict[str, Any]:
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)


tokenized_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=dataset["train"].column_names
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=5000,
    save_total_limit=2,
    logging_steps=500,
    evaluation_strategy="steps",
    eval_steps=5000,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=1000,
    gradient_accumulation_steps=1,
    fp16=True,  # 如果硬件支持，使用混合精度训练
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets.get("validation"),
    data_collator=data_collator,
)

trainer.train()
