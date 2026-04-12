#!/usr/bin/env python3
"""
Manual Validation Re-run with Diagnostics
Helps identify why validation metrics are zero
"""

from pathlib import Path
from ultralytics import YOLO
import json

def run_validation_diagnostic():
    """Re-run validation with detailed diagnostics"""
    
    project_root = Path("/kaggle/working/project")
    models_dir = project_root / "training_results" / "models"
    
    print("\n" + "="*80)
    print("MANUAL VALIDATION DIAGNOSTIC")
    print("="*80)
    
    for dataset in ["human-footprint", "vehicle"]:
        print(f"\n\n{'='*80}")
        print(f"DATASET: {dataset}")
        print(f"{'='*80}")
        
        # Find latest model
        search_pattern = f"{dataset}_*"
        runs = sorted(list(models_dir.glob(search_pattern)))
        
        if not runs:
            print(f"❌ No model runs found for {dataset}")
            continue
        
        latest_run = runs[-1]
        model_path = latest_run / "weights" / "best.pt"
        data_yaml = project_root / dataset / "data.yaml"
        
        print(f"\nModel: {model_path.name}")
        print(f"Data: {data_yaml}")
        
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            continue
        
        if not data_yaml.exists():
            print(f"❌ Data YAML not found: {data_yaml}")
            continue
        
        # Load model
        print("\n[1] Loading model...")
        try:
            model = YOLO(str(model_path))
            print(f"   ✅ Model loaded")
            print(f"   Task: {model.task}")
            print(f"   Model type: {type(model.model).__name__}")
        except Exception as e:
            print(f"   ❌ Failed to load model: {e}")
            continue
        
        # Read data.yaml
        print("\n[2] Reading data config...")
        try:
            with open(data_yaml) as f:
                config_content = f.read()
            print(f"   ✅ Config loaded")
            print("   Content:")
            for line in config_content.split('\n'):
                if line.strip():
                    print(f"     {line}")
        except Exception as e:
            print(f"   ❌ Failed to read config: {e}")
        
        # Check validation directory
        print("\n[3] Checking validation data...")
        valid_dir = project_root / dataset / "valid"
        valid_images = valid_dir / "images"
        valid_labels = valid_dir / "labels"
        
        if valid_images.exists():
            img_count = len(list(valid_images.glob("*")))
            print(f"   ✅ Valid images: {img_count}")
        else:
            print(f"   ❌ Valid images dir not found")
        
        if valid_labels.exists():
            label_count = len(list(valid_labels.glob("*.txt")))
            print(f"   ✅ Valid labels: {label_count}")
        else:
            print(f"   ❌ Valid labels dir not found")
        
        # Test inference on first image
        print("\n[4] Testing inference on single image...")
        try:
            test_image = list(valid_images.glob("*.jpg"))[0] if valid_images.exists() else None
            
            if test_image:
                print(f"   Image: {test_image.name}")
                results = model.predict(str(test_image), conf=0.1, verbose=False)
                
                num_boxes = len(results[0].boxes) if results else 0
                print(f"   ✅ Inference successful")
                print(f"   Boxes detected: {num_boxes}")
                
                if num_boxes > 0:
                    print(f"   Confidences: {[f'{c:.4f}' for c in results[0].boxes.conf.tolist()[:5]]}")
                    if len(results[0].boxes.conf) > 5:
                        print(f"   ... and {len(results[0].boxes.conf) - 5} more")
        except Exception as e:
            print(f"   ❌ Inference failed: {e}")
        
        # Run validation with different confidence thresholds
        print("\n[5] Running validation with different confidence thresholds...")
        
        for conf_thresh in [0.5, 0.25, 0.1]:
            print(f"\n   Confidence threshold: {conf_thresh}")
            
            try:
                metrics = model.val(
                    data=str(data_yaml),
                    conf=conf_thresh,
                    verbose=False,
                    device=0
                )
                
                print(f"      mAP50: {metrics.box.map50:.4f}")
                print(f"      AP: {metrics.box.ap[:5]}")  # First 5 classes
                
            except Exception as e:
                print(f"      ❌ Validation failed: {str(e)[:100]}")
        
        # Check model output shape
        print("\n[6] Checking model output...")
        try:
            # Create dummy input
            import torch
            dummy_input = torch.randn(1, 3, 640, 640).to(model.device)
            output = model.model(dummy_input)
            
            print(f"   ✅ Model forward pass successful")
            if isinstance(output, tuple):
                print(f"   Output: tuple with {len(output)} elements")
                for i, o in enumerate(output):
                    if hasattr(o, 'shape'):
                        print(f"     [{i}] shape: {o.shape}")
            else:
                print(f"   Output shape: {output.shape}")
                
        except Exception as e:
            print(f"   ⚠️  Forward pass check failed: {str(e)[:100]}")
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    print("\n💡 INTERPRETATION:")
    print("   • If boxes detected at low conf: model learned! Lower confidence threshold")
    print("   • If NO boxes at any conf: model didn't learn, retraining needed")
    print("   • If inference fails: data format incompatibility")

if __name__ == '__main__':
    run_validation_diagnostic()
