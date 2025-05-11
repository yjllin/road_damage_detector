from pathlib import Path
import cv2
import os
root_dir = "./database"
dataset_names = os.listdir(root_dir)
for dataset_name in dataset_names:
    dataset_dir = os.path.join(root_dir, dataset_name)
    dataset_dir = os.path.join(dataset_dir, "yolo")
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")
    train_count = 0
    val_count = 0
    for img_path in Path(train_dir).glob("*.jpg"):
        img = cv2.imread(str(img_path))
        if img is None:
            train_count += 1
    for img_path in Path(val_dir).glob("*.jpg"):
        img = cv2.imread(str(img_path))
        if img is None:
            val_count += 1
    print(f"{dataset_name} 损坏的图像数量: {train_count + val_count}")  