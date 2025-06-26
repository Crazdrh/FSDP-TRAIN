import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ====== CONFIGURATION ======
MODEL_NAME = "microsoft/Phi-4-reasoning-plus"  # Change to your model or local path
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,          # Use 8-bit quantization
    llm_int8_threshold=6.0,     # Standard threshold
    llm_int8_has_fp16_weight=False, # For older GPUs
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cuda",              # Auto-places model on GPU/CPU
    quantization_config=bnb_config, # Pass the config here!
    torch_dtype=torch.float16,      # Needed for some models
)
model.eval()

def chat(user_input):
    # For instruct models
    prompt = f"### User: {user_input}\n### Assistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Assistant:" in output_text:
        reply = output_text.split("### Assistant:")[-1].strip()
    else:
        reply = output_text
    return reply

if __name__ == "__main__":
    while True:
        user_input = "C:/Users/Hayden/OneDrive/Desktop/gpu install pytorch.txt"
        if user_input.lower() in {"exit", "quit"}:
            break
        reply = chat(user_input)
        print("Assistant:", reply)
