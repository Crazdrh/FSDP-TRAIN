import os
import shutil

src_dir = 'path to input'  # Path to your directory with the files
dst_root = 'path to output'  # Path where the new dirs will be created

num_dirs = 20

# Define allowed extensions
ALLOWED_EXTENSIONS = ('.txt', '.json', '.yaml', '.yml', '.md', '.csv')  # Add more as needed

# List all allowed files in source dir
all_files = [f for f in os.listdir(src_dir) if f.lower().endswith(ALLOWED_EXTENSIONS)]
all_files.sort()  # Optional: to make distribution deterministic

files_per_dir = len(all_files) // num_dirs
remainder = len(all_files) % num_dirs

start = 0
for i in range(num_dirs):
    # Create destination subdir
    dst_dir = os.path.join(dst_root, f'split_{i:02d}')
    os.makedirs(dst_dir, exist_ok=True)

    # Compute how many files for this dir (spread remainder over first ones)
    end = start + files_per_dir + (1 if i < remainder else 0)

    # Move files
    for fname in all_files[start:end]:
        shutil.move(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))

    print(f'Moved {end - start} files to {dst_dir}')
    start = end
