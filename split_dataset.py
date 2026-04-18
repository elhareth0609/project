#!/usr/bin/env python3
"""
YOLO 11 Dataset Split Script
Combines and splits multiple YOLO-format datasets into train/val/test sets (70/15/15)
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
        return False, []
    if not labels_dir.exists():
        print(f"❌ Labels directory not found: {labels_dir}")
        return False, []
    
    # Get all images (support common image formats)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    all_images = [f for f in os.listdir(images_dir) 
                  if Path(f).suffix.lower() in image_extensions]
    
    print(f"📊 Found {len(all_images)} images")
    
    if len(all_images) == 0:
        print("❌ No images found!")
        return False, []
    
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
    
    # Create valid and test directories and copy files
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
            
            shutil.move(str(src_img), str(dst_img))
            shutil.move(str(src_label), str(dst_label))
        
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
    return True, valid_images


def combine_datasets(output_path, source_datasets):
    """
    Combine multiple YOLO datasets into one, remapping class IDs if needed.
    
    Args:
        output_path: Path where combined dataset will be created
        source_datasets: List of tuples (source_path, class_mapping)
                        class_mapping: dict mapping old class ids to class names
    """
    output_path = Path(output_path)
    output_train_img = output_path / 'train' / 'images'
    output_train_lbl = output_path / 'train' / 'labels'
    output_val_img = output_path / 'val' / 'images'
    output_val_lbl = output_path / 'val' / 'labels'
    output_test_img = output_path / 'test' / 'images'
    output_test_lbl = output_path / 'test' / 'labels'
    
    # Create directories
    for d in [output_train_img, output_train_lbl, output_val_img, output_val_lbl, 
              output_test_img, output_test_lbl]:
        d.mkdir(parents=True, exist_ok=True)
    
    all_classes = {}
    class_mapping = {}  # Maps (dataset_idx, old_id) -> new_id
    next_class_id = 0
    
    print(f"\n{'='*60}")
    print(f"Combining datasets into: {output_path.name}")
    print(f"{'='*60}\n")
    
    for dataset_idx, (source_path, class_names) in enumerate(source_datasets):
        source_path = Path(source_path)
        print(f"Processing dataset {dataset_idx + 1}: {source_path.name}")
        
        # Build class mapping for this dataset
        for old_id, class_name in enumerate(class_names):
            if class_name not in all_classes:
                all_classes[class_name] = next_class_id
                next_class_id += 1
            class_mapping[(dataset_idx, old_id)] = all_classes[class_name]
        
        # Copy and remap labels
        for split, (src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir) in [
            ('train', (source_path / 'train' / 'images', source_path / 'train' / 'labels',
                      output_train_img, output_train_lbl)),
            ('val', (source_path / 'valid' / 'images', source_path / 'valid' / 'labels',
                    output_val_img, output_val_lbl)),
            ('test', (source_path / 'test' / 'images', source_path / 'test' / 'labels',
                     output_test_img, output_test_lbl))
        ]:
            if not src_img_dir.exists():
                continue
                
            for img_file in src_img_dir.iterdir():
                if img_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}:
                    # Copy image
                    unique_name = f"{source_path.name}_{img_file.name}"
                    shutil.copy(str(img_file), str(dst_img_dir / unique_name))
                    
                    # Process and copy label
                    label_file = src_lbl_dir / (img_file.stem + '.txt')
                    if label_file.exists():
                        with open(label_file, 'r') as f:
                            lines = f.readlines()
                        
                        # Remap class IDs
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
    
    # Print class mapping
    print(f"\n📋 Final Class Mapping:")
    for class_name, class_id in sorted(all_classes.items(), key=lambda x: x[1]):
        print(f"   {class_id}: {class_name}")
    
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
    
    print(f"\n✅ Combined dataset created at: {output_path}")
    print(f"📝 data.yaml created with {len(all_classes)} classes")
    print(f"\n{'='*60}\n")
    
    return all_classes


def main():
    """Split individual datasets and combine them"""
    project_root = Path(__file__).parent
    
    # Step 1: Split individual datasets
    datasets_to_split = ['animals', 'humain']
    split_results = {}
    
    for dataset_name in datasets_to_split:
        dataset_path = project_root / dataset_name
        if dataset_path.exists():
            success, images = split_dataset(dataset_path)
            split_results[dataset_name] = (success, len(images) if success else 0)
        else:
            print(f"❌ Dataset not found: {dataset_path}")
            split_results[dataset_name] = (False, 0)
    
    # Step 2: Combine datasets
    combined_path = project_root / 'dataset_combined'
    source_datasets = [
        (project_root / 'animals', ['animal_footprint']),
        (project_root / 'humain', ['human_footprint'])
    ]
    
    combine_datasets(combined_path, source_datasets)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\n✅ Individual Dataset Splits:")
    for dataset, (success, count) in split_results.items():
        status = f"✅ {count} images" if success else "❌ Failed"
        print(f"   {dataset}: {status}")
    
    print(f"\n✅ Combined Dataset:")
    print(f"   Path: {combined_path}")
    print(f"   Ready for YOLO 11 training!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
