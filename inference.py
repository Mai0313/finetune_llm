from rich.console import Console
from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig

console = Console()

model_id = "meta-llama/Llama-3.2-1B"

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
tokenizer = LlamaTokenizer.from_pretrained(model_id)
model = LlamaForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)

inputs = tokenizer("提供一段文字說明：如何準備咖啡", return_tensors="pt").input_ids
outputs = model.generate(inputs, max_new_tokens=50)
console.print(tokenizer.decode(outputs[0], skip_special_tokens=True))
