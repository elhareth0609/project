#!/usr/bin/env python3
"""
Diagnostic script to investigate zero validation metrics
"""

import json
import os
from pathlib import Path
from ultralytics import YOLO

def diagnose_validation():
    """Diagnose validation issues"""
    
    project_root = Path("/kaggle/working/project")
    results_dir = project_root / "training_results"
    models_dir = results_dir / "models"
    
    print("="*80)
    print("VALIDATION DIAGNOSTICS")
    print("="*80)
    
    # 1. Check trained models exist
    print("\n1️⃣  CHECKING TRAINED MODELS")
    print("-"*80)
    
    for dataset in ["human-footprint", "vehicle"]:
        # Find the latest run
        search_pattern = f"{dataset}_*"
        runs = list(models_dir.glob(search_pattern))
        
        if runs:
            latest_run = sorted(runs)[-1]
            model_path = latest_run / "weights" / "best.pt"
            
            print(f"\n📦 {dataset}:")
            print(f"   Model path: {model_path}")
            print(f"   Exists: {model_path.exists()}")
            
            if model_path.exists():
                size = model_path.stat().st_size / (1024*1024)
                print(f"   Size: {size:.1f} MB")
                
                # Try loading model
                try:
                    model = YOLO(str(model_path))
                    print(f"   ✅ Model loaded successfully")
                    print(f"   Model type: {model.model.__class__.__name__}")
                except Exception as e:
                    print(f"   ❌ Error loading model: {e}")
    
    # 2. Check validation data
    print("\n\n2️⃣  CHECKING VALIDATION DATA")
    print("-"*80)
    
    for dataset in ["human-footprint", "vehicle"]:
        data_yaml = project_root / dataset / "data.yaml"
        valid_images_dir = project_root / dataset / "valid" / "images"
        valid_labels_dir = project_root / dataset / "valid" / "labels"
        
        print(f"\n📁 {dataset}:")
        print(f"   data.yaml: {data_yaml.exists()}")
        print(f"   Valid images dir: {valid_images_dir.exists()}")
        print(f"   Valid labels dir: {valid_labels_dir.exists()}")
        
        if valid_images_dir.exists():
            img_count = len(list(valid_images_dir.glob("*")))
            print(f"   ✓ Images: {img_count}")
        
        if valid_labels_dir.exists():
            label_count = len(list(valid_labels_dir.glob("*.txt")))
            print(f"   ✓ Labels: {label_count}")
        
        # Read data.yaml
        if data_yaml.exists():
            with open(data_yaml, 'r') as f:
                print(f"\n   data.yaml content:")
                for line in f:
                    print(f"     {line.rstrip()}")
    
    # 3. Test inference on a single image
    print("\n\n3️⃣  TESTING INFERENCE ON SINGLE IMAGE")
    print("-"*80)
    
    for dataset in ["human-footprint", "vehicle"]:
        search_pattern = f"{dataset}_*"
        runs = list(models_dir.glob(search_pattern))
        
        if runs:
            latest_run = sorted(runs)[-1]
            model_path = latest_run / "weights" / "best.pt"
            
            if model_path.exists():
                valid_images_dir = project_root / dataset / "valid" / "images"
                images = list(valid_images_dir.glob("*.jpg")) + list(valid_images_dir.glob("*.png"))
                
                if images:
                    test_image = images[0]
                    print(f"\n🖼️  {dataset}:")
                    print(f"   Test image: {test_image.name}")
                    
                    try:
                        model = YOLO(str(model_path))
                        results = model(str(test_image), verbose=False)
                        
                        print(f"   ✅ Inference successful")
                        print(f"   Detections: {len(results[0].boxes)}")
                        
                        if len(results[0].boxes) > 0:
                            print(f"   Box confidences: {results[0].boxes.conf.tolist()[:5]}")
                        
                    except Exception as e:
                        print(f"   ❌ Inference failed: {e}")
    
    # 4. Check validation metrics file
    print("\n\n4️⃣  CHECKING METRICS FILES")
    print("-"*80)
    
    metrics_file = results_dir / "metrics" / "validation_metrics.json"
    
    if metrics_file.exists():
        print(f"✅ Metrics file exists: {metrics_file}")
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            
        for dataset, values in metrics.items():
            print(f"\n📊 {dataset}:")
            for key, val in values.items():
                print(f"   {key}: {val}")
    else:
        print(f"❌ Metrics file not found: {metrics_file}")
    
    print("\n" + "="*80)
    print("DIAGNOSTICS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    diagnose_validation()
