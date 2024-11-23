from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
tokenizer = LlamaTokenizer.from_pretrained(model_id)

inputs = tokenizer("提供一段文字說明：如何準備咖啡？", return_tensors="pt").input_ids
outputs = model.generate(inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
