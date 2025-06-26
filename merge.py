import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# ==== USER CONFIG ====
BASE_MODEL_DIR = "/path/to/base/model"            # Example: "microsoft/Phi-3-mini-128k-instruct"
LORA_ADAPTER_DIR = "/path/to/lora/adapter_dir"    # Where you saved your trained LoRA adapters
MERGED_SAVE_DIR = "/path/to/output/merged_model"  # Directory to save merged model

# 1. Load base model and LoRA adapter
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_DIR, torch_dtype=torch.float16)
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, LORA_ADAPTER_DIR)

# 2. Merge LoRA weights into the model
print("Merging LoRA weights into base model...")
model = model.merge_and_unload()  # After this, model is plain Transformers!

# 3. Save merged model (you only need tokenizer if you want to include it)
print(f"Saving merged model to {MERGED_SAVE_DIR}")
model.save_pretrained(MERGED_SAVE_DIR)

# Optionally: save tokenizer too
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
tokenizer.save_pretrained(MERGED_SAVE_DIR)

print("Merge complete! You can now load the model from", MERGED_SAVE_DIR)
