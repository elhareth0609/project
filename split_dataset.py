#!/usr/bin/env python3
"""
YOLO Dataset Split Script
Splits YOLO-format datasets into train/valid/test sets (70/15/15)
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

def split_dataset(dataset_path, train_ratio=0.70, valid_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split YOLO dataset into train, valid, and test sets.
    
    Args:
        dataset_path: Path to dataset root (contains train/ subfolder)
        train_ratio: Proportion for training (default 0.70 = 70%)
        valid_ratio: Proportion for validation (default 0.15 = 15%)
        test_ratio: Proportion for testing (default 0.15 = 15%)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    dataset_path = Path(dataset_path)
    train_dir = dataset_path / "train"
    images_dir = train_dir / "images"
    labels_dir = train_dir / "labels"
    
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_path.name}")
    print(f"{'='*60}")
    
    # Validate directories exist
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return False
    if not labels_dir.exists():
        print(f"❌ Labels directory not found: {labels_dir}")
        return False
    
    # Get all images (support common image formats)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    all_images = [f for f in os.listdir(images_dir) 
                  if Path(f).suffix.lower() in image_extensions]
    
    print(f"📊 Found {len(all_images)} images")
    
    if len(all_images) == 0:
        print("❌ No images found!")
        return False
    
    # Verify labels exist for each image
    missing_labels = []
    valid_images = []
    
    for img in all_images:
        label_name = Path(img).stem + '.txt'
        label_path = labels_dir / label_name
        if label_path.exists():
            valid_images.append(img)
        else:
            missing_labels.append(img)
    
    if missing_labels:
        print(f"⚠️  {len(missing_labels)} images missing labels (skipping)")
    
    print(f"✅ {len(valid_images)} images with corresponding labels")
    
    # Shuffle images
    random.shuffle(valid_images)
    
    # Calculate split indices
    train_count = int(len(valid_images) * train_ratio)
    valid_count = int(len(valid_images) * valid_ratio)
    
    train_split = valid_images[:train_count]
    valid_split = valid_images[train_count:train_count + valid_count]
    test_split = valid_images[train_count + valid_count:]
    
    print(f"\n📈 Split ratios:")
    print(f"   Train: {len(train_split)} ({len(train_split)/len(valid_images)*100:.1f}%)")
    print(f"   Valid: {len(valid_split)} ({len(valid_split)/len(valid_images)*100:.1f}%)")
    print(f"   Test:  {len(test_split)} ({len(test_split)/len(valid_images)*100:.1f}%)")
    
    # Create valid and test directories and move files
    splits = {
        'valid': (dataset_path / 'valid', valid_split),
        'test': (dataset_path / 'test', test_split)
    }
    
    for split_name, (split_dir, image_list) in splits.items():
        split_images_dir = split_dir / 'images'
        split_labels_dir = split_dir / 'labels'
        
        # Create directories
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Move images and labels
        for img in image_list:
            src_img = images_dir / img
            dst_img = split_images_dir / img
            
            label_name = Path(img).stem + '.txt'
            src_label = labels_dir / label_name
            dst_label = split_labels_dir / label_name
            
            shutil.move(src_img, dst_img)
            shutil.move(src_label, dst_label)
        
        print(f"📁 Created {split_name} set: {split_images_dir}")
    
    # Update data.yaml if it exists
    data_yaml = dataset_path / 'data.yaml'
    if data_yaml.exists():
        print(f"\n📝 Updating {data_yaml.name}...")
        with open(data_yaml, 'r') as f:
            content = f.read()
        
        # Update paths to relative paths
        updated_content = content.replace(
            'train: ../train/images',
            'train: ./train/images'
        ).replace(
            'val: ../valid/images',
            'val: ./valid/images'
        ).replace(
            'test: ../test/images',
            'test: ./test/images'
        )
        
        with open(data_yaml, 'w') as f:
            f.write(updated_content)
        print("✅ data.yaml updated")
    
    print(f"\n{'='*60}\n")
    return True


def main():
    """Split both datasets"""
    project_root = Path("/kaggle/working/project")
    
    datasets = ['human-footprint', 'vehicle']
    results = {}
    
    for dataset_name in datasets:
        dataset_path = project_root / dataset_name
        if dataset_path.exists():
            success = split_dataset(dataset_path)
            results[dataset_name] = "✅ Success" if success else "❌ Failed"
        else:
            print(f"❌ Dataset not found: {dataset_path}")
            results[dataset_name] = "❌ Not found"
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for dataset, status in results.items():
        print(f"{dataset}: {status}")
    print("="*60)


if __name__ == '__main__':
    main()
