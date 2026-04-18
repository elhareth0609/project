#!/usr/bin/env python3
"""
YOLO 11 KAGGLE NOTEBOOK - Copy this to Kaggle Notebook
Complete pipeline for training YOLOv11 on footprint detection dataset
"""

# ============================================================
# SETUP & DEPENDENCIES
# ============================================================

# Install required packages
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    packages = [
        'ultralytics>=8.0.0',
        'pyyaml',
        'opencv-python',
        'numpy',
        'pandas'
    ]
    
    print("Installing dependencies...\n")
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
    print("\n✅ All dependencies installed!\n")

install_requirements()

# ============================================================
# IMPORTS
# ============================================================

import os
import shutil
import random
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
from ultralytics import YOLO

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# CELL 1: UPLOAD & ORGANIZE DATA
# ============================================================

print("\n" + "="*60)
print("STEP 1: PREPARE DATASET")
print("="*60 + "\n")

# Create working directory
working_dir = Path('/kaggle/working')
project_dir = working_dir / 'project'
project_dir.mkdir(exist_ok=True)

# Create dataset directories
dataset_dir = project_dir / 'dataset_combined'
dataset_dir.mkdir(exist_ok=True)

print("✅ Directories created")
print(f"Working Directory: {project_dir}")

# ============================================================
# CELL 2: SPLIT FUNCTION
# ============================================================

def split_dataset(dataset_path, train_ratio=0.70, valid_ratio=0.15, test_ratio=0.15, seed=42):
    """Split YOLO dataset"""
    random.seed(seed)
    
    dataset_path = Path(dataset_path)
    train_dir = dataset_path / "train"
    images_dir = train_dir / "images"
    labels_dir = train_dir / "labels"
    
    if not images_dir.exists() or not labels_dir.exists():
        return False, 0
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    all_images = [f for f in os.listdir(images_dir) 
                  if Path(f).suffix.lower() in image_extensions]
    
    valid_images = []
    for img in all_images:
        label_name = Path(img).stem + '.txt'
        if (labels_dir / label_name).exists():
            valid_images.append(img)
    
    random.shuffle(valid_images)
    
    train_count = int(len(valid_images) * train_ratio)
    valid_count = int(len(valid_images) * valid_ratio)
    
    train_split = valid_images[:train_count]
    valid_split = valid_images[train_count:train_count + valid_count]
    test_split = valid_images[train_count + valid_count:]
    
    # Create directories
    for split_name, image_list in [('valid', valid_split), ('test', test_split)]:
        split_dir = dataset_path / split_name
        split_images_dir = split_dir / 'images'
        split_labels_dir = split_dir / 'labels'
        
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)
        
        for img in image_list:
            src_img = images_dir / img
            dst_img = split_images_dir / img
            src_label = labels_dir / (Path(img).stem + '.txt')
            dst_label = split_labels_dir / src_label.name
            
            shutil.move(str(src_img), str(dst_img))
            shutil.move(str(src_label), str(dst_label))
    
    return True, len(valid_images)


# ============================================================
# CELL 3: COMBINE DATASETS
# ============================================================

def combine_datasets_kaggle(output_path, source_datasets):
    """Combine multiple YOLO datasets"""
    output_path = Path(output_path)
    
    # Create directories
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    all_classes = {}
    class_mapping = {}
    next_class_id = 0
    
    for dataset_idx, (source_path, class_names) in enumerate(source_datasets):
        source_path = Path(source_path)
        print(f"Processing: {source_path.name}")
        
        for old_id, class_name in enumerate(class_names):
            if class_name not in all_classes:
                all_classes[class_name] = next_class_id
                next_class_id += 1
            class_mapping[(dataset_idx, old_id)] = all_classes[class_name]
        
        # Copy files
        for split in ['train', 'val', 'test']:
            src_img_dir = source_path / ('valid' if split == 'val' else split) / 'images'
            src_lbl_dir = source_path / ('valid' if split == 'val' else split) / 'labels'
            dst_img_dir = output_path / split / 'images'
            dst_lbl_dir = output_path / split / 'labels'
            
            if not src_img_dir.exists():
                continue
            
            for img_file in src_img_dir.iterdir():
                if img_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}:
                    unique_name = f"{source_path.name}_{img_file.name}"
                    shutil.copy(str(img_file), str(dst_img_dir / unique_name))
                    
                    label_file = src_lbl_dir / (img_file.stem + '.txt')
                    if label_file.exists():
                        with open(label_file, 'r') as f:
                            lines = f.readlines()
                        
                        remapped_lines = []
                        for line in lines:
                            parts = line.strip().split()
                            if parts:
                                old_id = int(parts[0])
                                new_id = class_mapping[(dataset_idx, old_id)]
                                remapped_line = str(new_id) + ' ' + ' '.join(parts[1:]) + '\n'
                                remapped_lines.append(remapped_line)
                        
                        unique_label_name = f"{source_path.name}_{label_file.name}"
                        with open(dst_lbl_dir / unique_label_name, 'w') as f:
                            f.writelines(remapped_lines)
    
    # Create data.yaml
    data_yaml_content = f"""path: {output_path}
train: train/images
val: val/images
test: test/images

nc: {len(all_classes)}
names: {list(all_classes.keys())}
"""
    
    with open(output_path / 'data.yaml', 'w') as f:
        f.write(data_yaml_content)
    
    print(f"\n✅ Combined dataset with {len(all_classes)} classes")
    return all_classes


# ============================================================
# CELL 4: PREPARE DATA (Run this after uploading)
# ============================================================

print("\n" + "="*60)
print("STEP 2: PREPARE & COMBINE DATASETS")
print("="*60 + "\n")

# Assuming you've uploaded 'animals' and 'humain' folders to /kaggle/input/
# Copy them to working directory
input_dir = Path('/kaggle/input')

# Find the uploaded directories (may be in a subdirectory)
animals_src = None
humain_src = None

for path in input_dir.rglob('*/train/images'):
    if 'animal' in str(path).lower():
        animals_src = path.parent.parent
    if 'humain' in str(path).lower() or 'human' in str(path).lower():
        humain_src = path.parent.parent

if not animals_src or not humain_src:
    print("⚠️ Looking for dataset directories...")
    for item in input_dir.rglob('*'):
        print(item.relative_to(input_dir))

# Copy datasets
if animals_src:
    shutil.copytree(str(animals_src), str(project_dir / 'animals'), dirs_exist_ok=True)
    print("✅ Animals dataset copied")

if humain_src:
    shutil.copytree(str(humain_src), str(project_dir / 'humain'), dirs_exist_ok=True)
    print("✅ Humain dataset copied")

# Split individual datasets
print("\n📊 Splitting datasets...")
for dataset_name in ['animals', 'humain']:
    dataset_path = project_dir / dataset_name
    if dataset_path.exists():
        success, count = split_dataset(dataset_path)
        print(f"  {dataset_name}: {count} images ({'✅' if success else '❌'})")

# Combine datasets
print("\n🔄 Combining datasets...")
source_datasets = [
    (project_dir / 'animals', ['animal_footprint']),
    (project_dir / 'humain', ['human_footprint'])
]

classes = combine_datasets_kaggle(dataset_dir, source_datasets)

print(f"\n📋 Classes: {classes}")

# ============================================================
# CELL 5: LOAD DATASET CONFIG
# ============================================================

with open(dataset_dir / 'data.yaml', 'r') as f:
    data = yaml.safe_load(f)

print("\n" + "="*60)
print("DATASET CONFIG")
print("="*60)
print(f"Classes: {data['nc']}")
print(f"Class Names: {data['names']}")
print(f"Train: {data['train']}")
print(f"Val: {data['val']}")
print(f"Test: {data['test']}")
print("="*60)

# ============================================================
# CELL 6: TRAIN YOLO11
# ============================================================

print("\n" + "="*60)
print("STEP 3: TRAIN YOLO11")
print("="*60 + "\n")

# Training parameters
MODEL_SIZE = 'm'  # Options: n (nano), s (small), m (medium), l (large), x (xlarge)
EPOCHS = 50
BATCH_SIZE = 16  # Kaggle usually has 16GB GPU

print(f"Model: yolo11{MODEL_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Batch Size: {BATCH_SIZE}")
print()

# Load model
model = YOLO(f'yolo11{MODEL_SIZE}.pt')

# Train
results = model.train(
    data=str(dataset_dir / 'data.yaml'),
    epochs=EPOCHS,
    imgsz=640,
    batch=BATCH_SIZE,
    patience=10,
    save=True,
    device=0,  # Use GPU 0
    project='/kaggle/working',
    name='yolo11_footprint_detection',
    exist_ok=False,
    verbose=True,
    # Augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,
    translate=0.1,
    scale=0.5,
    flipud=0.5,
    fliplr=0.5,
    # Optimization
    optimizer='auto',
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
)

# ============================================================
# CELL 7: EVALUATE & VISUALIZE
# ============================================================

best_model_path = Path('/kaggle/working/yolo11_footprint_detection/weights/best.pt')

if best_model_path.exists():
    print("\n✅ Best model found!")
    
    # Load best model
    best_model = YOLO(str(best_model_path))
    
    # Validate
    print("\nValidating on val set...")
    val_results = best_model.val()
    
    print("\nMetrics:")
    print(f"mAP@50: {val_results.box.map50:.3f}")
    print(f"mAP@50-95: {val_results.box.map:.3f}")

# ============================================================
# CELL 8: TEST PREDICTIONS
# ============================================================

print("\n" + "="*60)
print("STEP 4: TEST PREDICTIONS")
print("="*60 + "\n")

test_images_dir = dataset_dir / 'test' / 'images'
test_images = list(test_images_dir.glob('*.jpg'))[:5]

if test_images:
    print(f"Running predictions on {len(test_images)} test images...\n")
    
    # Predict
    results = best_model.predict(
        source=test_images,
        conf=0.25,
        save=True,
        project='/kaggle/working',
        name='predictions',
        device=0
    )
    
    print("\n✅ Predictions complete!")

# ============================================================
# CELL 9: EXPORT RESULTS
# ============================================================

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

model_dir = Path('/kaggle/working/yolo11_footprint_detection')
print(f"\n📁 Results saved in: {model_dir}")
print(f"\n📊 Key files:")
print(f"   - Best Model: {model_dir / 'weights' / 'best.pt'}")
print(f"   - Last Model: {model_dir / 'weights' / 'last.pt'}")
print(f"   - Training Plots: {model_dir / 'results.png'}")
print(f"   - Confusion Matrix: {model_dir / 'confusion_matrix.png'}")

print("\n📝 To use the model:")
print("   from ultralytics import YOLO")
print("   model = YOLO('/kaggle/working/yolo11_footprint_detection/weights/best.pt')")
print("   results = model.predict(source='image.jpg', conf=0.25)")

print("\n" + "="*60)
