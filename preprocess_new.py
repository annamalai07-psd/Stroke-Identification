import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

# ==== USER CONFIG ====
input_root = "C:/CMCH/split and preprocess prgs/ps_8020"
output_root = "C:/CMCH/split and preprocess prgs/pp_8020"
img_size = (256, 256)
apply_brain_crop = False  # True if you want brain crop

# ==== Functions ====

def crop_brain(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(th)
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y+h, x:x+w]

def preprocess_image(path):
    img = cv2.imread(path)
    if img is None: 
        return None
    
    if apply_brain_crop:
        img = crop_brain(img)

    img = cv2.resize(img, img_size)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl,a,b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img

def preprocess_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_COLOR)
    if mask is None:
        return None

    # Extract red channel (binary mask stored here originally)
    red = mask[:,:,2]

    # Convert to pure binary mask (0 or 255)
    binary = np.where(red == 255, 255, 0).astype(np.uint8)

    # Resize using nearest-neighbor to preserve mask shape
    binary = cv2.resize(binary, img_size, interpolation=cv2.INTER_NEAREST)

    return binary
    

# ==== Pipeline ====
for split in ["train", "test"]:
    for category in ["normal", "stroke"]:

        if category == "normal":
            input_img_dir = Path(f"{input_root}/{split}/normal")
            output_img_dir = Path(f"{output_root}/{split}/normal")
            output_img_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n🟦 Processing Normal Images: {input_img_dir}")
            for img_file in tqdm(list(input_img_dir.glob("*.png"))):
                processed = preprocess_image(str(img_file))
                if processed is not None:
                    cv2.imwrite(str(output_img_dir / img_file.name), processed)

        else:  # stroke images
            input_img_dir = Path(f"{input_root}/{split}/stroke/png")
            input_mask_dir = Path(f"{input_root}/{split}/stroke/mask")

            output_img_dir = Path(f"{output_root}/{split}/stroke/png")
            output_mask_dir = Path(f"{output_root}/{split}/stroke/mask")
            output_img_dir.mkdir(parents=True, exist_ok=True)
            output_mask_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n🟧 Processing Stroke Images: {input_img_dir}")
            for img_file in tqdm(list(input_img_dir.glob("*.png"))):
                processed = preprocess_image(str(img_file))
                if processed is not None:
                    cv2.imwrite(str(output_img_dir / img_file.name), processed)

            print(f"\n🟥 Processing Stroke Masks: {input_mask_dir}")
            for mask_file in tqdm(list(input_mask_dir.glob("*.png"))):
                processed_mask = preprocess_mask(str(mask_file))
                if processed_mask is not None:
                    cv2.imwrite(str(output_mask_dir / mask_file.name), processed_mask)

print("\n✅ Preprocessing Completed Successfully!")
