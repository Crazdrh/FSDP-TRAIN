from datasets import Dataset, DatasetDict
import os
import json

try:
    import yaml
except ImportError:
    yaml = None  # Only use if installed

def extract_strings(obj):
    # Recursively extract all string values from any nested structure
    strings = []
    if isinstance(obj, str):
        strings.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            strings.extend(extract_strings(v))
    elif isinstance(obj, list):
        for item in obj:
            strings.extend(extract_strings(item))
    return strings

# Path to your folder of data files
data_folder = "path"   # CHANGE THIS TO YOUR FOLDER

data = []
for fname in os.listdir(data_folder):
    path = os.path.join(data_folder, fname)
    ext = fname.lower().split('.')[-1]

    try:
        if ext == "txt":
            with open(path, "r", encoding="utf-8") as f:
                data.append({"text": f.read()})

        elif ext == "json":
            with open(path, "r", encoding="utf-8") as f:
                j = json.load(f)
                for s in extract_strings(j):
                    data.append({"text": s})

        elif ext in ("yaml", "yml") and yaml is not None:
            with open(path, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f)
                for s in extract_strings(y):
                    data.append({"text": s})

        # Add other file types here if needed, or default to reading as text
        else:
            # For now, try to read as text (ignore binary, etc.)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data.append({"text": f.read()})
            except Exception as e:
                print(f"Skipping {fname}: {e}")

    except Exception as e:
        print(f"Error reading {fname}: {e}")

# Create Hugging Face Dataset
hf_dataset = Dataset.from_list(data)
print(hf_dataset)

# Save the dataset in Hugging Face Arrow format
save_dir = "path output"   # CHANGE if you want
hf_dataset.save_to_disk(save_dir)

print(f"Saved HF dataset at {save_dir}")
