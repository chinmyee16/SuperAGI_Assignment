
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 125M model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Sample input prompt
prompt = "Once upon a time"

# Tokenize and convert to tensor
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Generate output
with torch.no_grad():
    output = model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:", generated_text)
