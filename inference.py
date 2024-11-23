from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig

model_id = "path/to/llama-3.2-1b"

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
tokenizer = LlamaTokenizer.from_pretrained(model_id)
model = LlamaForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)

inputs = tokenizer("提供一段文字說明：如何準備咖啡？", return_tensors="pt").input_ids
outputs = model.generate(inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
