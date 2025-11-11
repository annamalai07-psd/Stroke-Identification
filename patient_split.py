import os
import shutil
import random

# === INPUT PATHS ===
normal_folder = "C:/CMCH/dataset2/normal"
stroke_img_folder = "C:/CMCH/dataset2/stroke/png"
stroke_mask_folder = "C:/CMCH/dataset2/stroke/mask"

# === OUTPUT PATH ===
output = "C:/CMCH/split and preprocess prgs/ps_8020"

train_ratio = 0.8  # 80% train, 20% test

# Create output directories
folders = [
    "train/normal", "test/normal",
    "train/stroke/png", "train/stroke/mask",
    "test/stroke/png", "test/stroke/mask"
]
for f in folders:
    os.makedirs(os.path.join(output, f), exist_ok=True)

def extract_patients(file_list):
    return list(set([f.split("_")[0] for f in file_list]))

# === NORMAL PATIENT SPLIT ===
normal_files = os.listdir(normal_folder)
normal_patients = extract_patients(normal_files)
random.shuffle(normal_patients)

n_train = int(len(normal_patients) * train_ratio)
normal_train = set(normal_patients[:n_train])
normal_test = set(normal_patients[n_train:])

# === STROKE PATIENT SPLIT ===
stroke_files = os.listdir(stroke_img_folder)
stroke_patients = extract_patients(stroke_files)
random.shuffle(stroke_patients)

s_train = int(len(stroke_patients) * train_ratio)
stroke_train = set(stroke_patients[:s_train])
stroke_test = set(stroke_patients[s_train:])

def copy_normal(split, patients):
    dst = os.path.join(output, f"{split}/normal")
    for f in normal_files:
        patient = f.split("_")[0]
        if patient in patients:
            shutil.copy(os.path.join(normal_folder, f), os.path.join(dst, f))

def copy_stroke(split, patients):
    img_dst = os.path.join(output, f"{split}/stroke/png")
    mask_dst = os.path.join(output, f"{split}/stroke/mask")
    for f in stroke_files:
        patient = f.split("_")[0]
        if patient in patients:
            shutil.copy(os.path.join(stroke_img_folder, f), os.path.join(img_dst, f))
            shutil.copy(os.path.join(stroke_mask_folder, f), os.path.join(mask_dst, f))

# Copy NORMAL
copy_normal("train", normal_train)
copy_normal("test", normal_test)

# Copy STROKE
copy_stroke("train", stroke_train)
copy_stroke("test", stroke_test)

print("✅ Patient-level train/test split completed.")
print(f"Normal Train: {len(normal_train)}, Test: {len(normal_test)}")
print(f"Stroke Train: {len(stroke_train)}, Test: {len(stroke_test)}")

