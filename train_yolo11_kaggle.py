#!/usr/bin/env python3
"""
YOLO 11 Training Script for Kaggle
Trains YOLOv11 model on combined footprint dataset
"""

import os
import sys
from pathlib import Path
import yaml

try:
    from ultralytics import YOLO
    import torch
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install ultralytics torch torchvision")
    sys.exit(1)


def check_environment():
    """Check if running on Kaggle or local"""
    is_kaggle = '/kaggle' in os.getcwd()
    is_gpu_available = torch.cuda.is_available()
    
    print("\n" + "="*60)
    print("ENVIRONMENT CHECK")
    print("="*60)
    print(f"Running on Kaggle: {is_kaggle}")
    print(f"GPU Available: {is_gpu_available}")
    if is_gpu_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch Version: {torch.__version__}")
    print("="*60 + "\n")
    
    return is_kaggle, is_gpu_available


def setup_paths(is_kaggle=False):
    """Setup directory paths based on environment"""
    if is_kaggle:
        project_root = Path("/kaggle/working/project")
        output_dir = Path("/kaggle/working/runs")
    else:
        project_root = Path(__file__).parent
        output_dir = project_root / "runs"
    
    dataset_path = project_root / "dataset_combined"
    data_yaml = dataset_path / "data.yaml"
    
    print(f"Project Root: {project_root}")
    print(f"Dataset Path: {dataset_path}")
    print(f"Output Directory: {output_dir}")
    
    # Verify dataset exists
    if not data_yaml.exists():
        print(f"\n❌ ERROR: data.yaml not found at {data_yaml}")
        print("Run split_dataset.py first to prepare the dataset")
        return None, None, None, None
    
    return project_root, dataset_path, data_yaml, output_dir


def load_data_yaml(data_yaml):
    """Load and display data.yaml contents"""
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    print("\n" + "="*60)
    print("DATASET CONFIGURATION")
    print("="*60)
    print(f"Classes: {data.get('nc', 0)}")
    print(f"Class Names: {data.get('names', [])}")
    print(f"Train Path: {data.get('train', 'N/A')}")
    print(f"Val Path: {data.get('val', 'N/A')}")
    print(f"Test Path: {data.get('test', 'N/A')}")
    print("="*60 + "\n")
    
    return data


def train_model(data_yaml, output_dir, model_size='m', epochs=50, batch_size=16, device=None):
    """
    Train YOLOv11 model
    
    Args:
        data_yaml: Path to data.yaml
        output_dir: Where to save results
        model_size: Model size (n=nano, s=small, m=medium, l=large, x=extra-large)
        epochs: Number of training epochs
        batch_size: Batch size
        device: Device to use (0 for GPU, 'cpu' for CPU)
    """
    
    print("="*60)
    print("TRAINING YOLO11")
    print("="*60)
    print(f"Model Size: yolo11{model_size}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Device: {device}")
    print("="*60 + "\n")
    
    # Load YOLOv11 model
    model_name = f'yolo11{model_size}.pt'
    print(f"Loading model: {model_name}...")
    model = YOLO(model_name)
    
    # Train the model
    print("\n🚀 Starting training...\n")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        patience=10,  # Early stopping patience
        save=True,
        save_period=5,
        device=device if device else (0 if torch.cuda.is_available() else 'cpu'),
        project=str(output_dir),
        name='yolo11_footprint',
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
        mosaic=1.0,
        # Optimization
        optimizer='auto',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
    )
    
    return results


def evaluate_model(data_yaml, model_path, output_dir):
    """
    Evaluate trained model on test set
    
    Args:
        data_yaml: Path to data.yaml
        model_path: Path to trained model weights
        output_dir: Where to save results
    """
    print("\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60 + "\n")
    
    model = YOLO(str(model_path))
    
    # Validate on val set
    print("Validating on validation set...")
    val_results = model.val()
    
    return val_results


def predict_sample(model_path, output_dir):
    """Test model with a sample prediction"""
    print("\n" + "="*60)
    print("RUNNING SAMPLE PREDICTIONS")
    print("="*60 + "\n")
    
    model = YOLO(str(model_path))
    
    # Try to find a test image
    dataset_dir = Path(__file__).parent / "dataset_combined"
    test_img_dir = dataset_dir / "test" / "images"
    
    if test_img_dir.exists():
        test_images = list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png"))
        if test_images:
            print(f"Testing on {len(test_images[:5])} sample images...\n")
            results = model.predict(
                source=test_images[:5],
                conf=0.25,
                save=True,
                project=str(output_dir),
                name='predictions'
            )
            print(f"\n✅ Predictions saved!")


def main():
    """Main training pipeline"""
    
    # Check environment
    is_kaggle, has_gpu = check_environment()
    
    # Setup paths
    project_root, dataset_path, data_yaml, output_dir = setup_paths(is_kaggle)
    if not data_yaml:
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset config
    data = load_data_yaml(data_yaml)
    
    # Training parameters
    MODEL_SIZE = 'm'  # Options: n, s, m, l, x (nano to extra-large)
    EPOCHS = 50
    BATCH_SIZE = 16 if has_gpu else 8  # Smaller batch for CPU
    
    # Train model
    results = train_model(
        data_yaml=data_yaml,
        output_dir=output_dir,
        model_size=MODEL_SIZE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    # Get best model path
    best_model = output_dir / 'yolo11_footprint' / 'weights' / 'best.pt'
    
    if best_model.exists():
        print(f"\n✅ Best model saved: {best_model}")
        
        # Evaluate model
        evaluate_model(data_yaml, best_model, output_dir)
        
        # Run predictions on samples
        predict_sample(best_model, output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir / 'yolo11_footprint'}")
    print(f"Best Model: {best_model}")
    print("\nTo use the model:")
    print(f"  from ultralytics import YOLO")
    print(f"  model = YOLO('{best_model}')")
    print(f"  results = model.predict(source='image.jpg')")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
