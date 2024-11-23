from datasets import load_dataset
from transformers import (
    LlamaTokenizer,
    Seq2SeqTrainer,
    LlamaForCausalLM,
    BitsAndBytesConfig,
    Seq2SeqTrainingArguments,
)

dataset = load_dataset("hugfaceguy0001/retarded_bar", "question")

# 載入 Tokenizer 和模型
model_id = "path/to/llama-3.2-1b"

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
tokenizer = LlamaTokenizer.from_pretrained(model_id)
model = LlamaForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)


def preprocess_function(examples: list[dict[str, str]]) -> dict[str, list]:
    inputs = ["[INST] " + ex["instruction"] + " [SEP] " + ex["input"] for ex in examples]
    outputs = [ex["output"] for ex in examples]
    return tokenizer(inputs, text_target=outputs, max_length=512, truncation=True)


tokenized_dataset = dataset.map(preprocess_function, batched=True)

training_args = Seq2SeqTrainingArguments(
    output_dir="./llama-finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    logging_dir="./logs",
    save_total_limit=3,
    save_steps=500,
    fp16=True,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model, args=training_args, train_dataset=tokenized_dataset["train"], tokenizer=tokenizer
)

trainer.train()
