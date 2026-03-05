import os
import random
import shutil

SOURCE_DIR = "data/all_human"
TRAIN_DIR = "data/train/human"
VAL_DIR = "data/val/human"

SPLIT_RATIO = 0.8  # 80% train, 20% val

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

files = [
    f for f in os.listdir(SOURCE_DIR)
    if f.lower().endswith(".wav")
]

if len(files) == 0:
    raise RuntimeError("No human audio files found!")

random.shuffle(files)

split_index = int(len(files) * SPLIT_RATIO)

train_files = files[:split_index]
val_files = files[split_index:]

for f in train_files:
    shutil.copy(
        os.path.join(SOURCE_DIR, f),
        os.path.join(TRAIN_DIR, f)
    )

for f in val_files:
    shutil.copy(
        os.path.join(SOURCE_DIR, f),
        os.path.join(VAL_DIR, f)
    )

print(f"Total human files: {len(files)}")
print(f"Train: {len(train_files)} | Val: {len(val_files)}")
