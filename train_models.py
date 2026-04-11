#!/usr/bin/env python3
"""
YOLOv8 Training Script for Footprint and Vehicle Detection
Trains models and generates comprehensive metrics
"""

import os
from pathlib import Path
from datetime import datetime
import json
from ultralytics import YOLO
import torch


class ModelTrainer:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.results_dir = self.project_root / 'training_results'
        self.results_dir.mkdir(exist_ok=True)
        self.training_log = []
        
    def log_event(self, message):
        """Log training events"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        self.training_log.append(log_message)
    
    def train_model(self, dataset_name, data_yaml_path, epochs=50, imgsz=640, device=0):
        """
        Train YOLOv8 model
        
        Args:
            dataset_name: Name of dataset (human-footprint, vehicle)
            data_yaml_path: Path to data.yaml
            epochs: Number of training epochs
            imgsz: Image size
            device: GPU device (0 for first GPU, -1 for CPU)
        """
        self.log_event(f"\n{'='*60}")
        self.log_event(f"Starting training: {dataset_name}")
        self.log_event(f"{'='*60}\n")
        
        # Validate data.yaml exists
        if not Path(data_yaml_path).exists():
            self.log_event(f"❌ data.yaml not found: {data_yaml_path}")
            return None
        
        try:
            # Initialize YOLOv8 model (nano for small datasets)
            self.log_event(f"📦 Loading YOLOv8 model...")
            model = YOLO('yolov8n.pt')  # nano model for smaller dataset
            
            # Check GPU availability
            if device != -1:
                if torch.cuda.is_available():
                    self.log_event(f"✅ GPU available: {torch.cuda.get_device_name(device)}")
                else:
                    self.log_event(f"⚠️ GPU not available, using CPU")
                    device = -1
            else:
                self.log_event(f"🖥️ Using CPU for training")
            
            # Create run name with timestamp
            run_name = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Train model
            self.log_event(f"🚀 Training started ({epochs} epochs, imgsz={imgsz})...")
            results = model.train(
                data=str(data_yaml_path),
                epochs=epochs,
                imgsz=imgsz,
                device=device,
                project=str(self.results_dir / 'models'),
                name=run_name,
                save=True,
                patience=5,  # Early stopping
                batch=16,
                workers=4,
                verbose=True,
                augment=True,
                mosaic=1.0,
                flipud=0.5,
                fliplr=0.5,
                degrees=10,
                translate=0.1,
                scale=0.5,
                perspective=0.0,
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
            )
            
            self.log_event(f"\n✅ Training completed!")
            self.log_event(f"📁 Results saved to: {self.results_dir / 'models' / run_name}")
            
            return {
                'dataset': dataset_name,
                'model_path': model.model_name,
                'results_dir': str(self.results_dir / 'models' / run_name),
                'results': results,
                'run_name': run_name
            }
            
        except Exception as e:
            self.log_event(f"❌ Training failed: {str(e)}")
            return None
    
    def validate_model(self, model_path, data_yaml_path):
        """Validate trained model"""
        try:
            self.log_event(f"\n📊 Validating model: {model_path}")
            model = YOLO(model_path)
            metrics = model.val(data=str(data_yaml_path))
            return metrics
        except Exception as e:
            self.log_event(f"❌ Validation failed: {str(e)}")
            return None
    
    def save_log(self):
        """Save training log to file"""
        log_file = self.results_dir / 'training_log.txt'
        with open(log_file, 'w') as f:
            f.write('\n'.join(self.training_log))
        self.log_event(f"\n📝 Log saved: {log_file}")


def main():
    print("\n" + "="*70)
    print(" "*20 + "YOLOV8 DETECTION MODEL TRAINER")
    print("="*70 + "\n")
    
    project_root = Path("c:/Users/elhareth/Downloads/project")
    trainer = ModelTrainer(project_root)
    
    datasets = {
        'human-footprint': project_root / 'human-footprint' / 'data.yaml',
        'vehicle': project_root / 'vehicle' / 'data.yaml'
    }
    
    trained_models = {}
    
    # Train both models
    for dataset_name, data_yaml in datasets.items():
        if data_yaml.exists():
            result = trainer.train_model(
                dataset_name=dataset_name,
                data_yaml_path=data_yaml,
                epochs=50,
                imgsz=640,
                device=0 if torch.cuda.is_available() else -1
            )
            if result:
                trained_models[dataset_name] = result
        else:
            trainer.log_event(f"❌ Config not found: {data_yaml}")
    
    # Save training log
    trainer.save_log()
    
    # Summary
    trainer.log_event(f"\n{'='*60}")
    trainer.log_event("TRAINING SUMMARY")
    trainer.log_event(f"{'='*60}")
    for dataset, info in trained_models.items():
        trainer.log_event(f"✅ {dataset}: Complete")
        trainer.log_event(f"   Run: {info['run_name']}")
        trainer.log_event(f"   Path: {info['results_dir']}")
    trainer.log_event(f"{'='*60}\n")
    
    # Save trained model info
    model_info = {
        'timestamp': datetime.now().isoformat(),
        'trained_models': trained_models
    }
    
    with open(trainer.results_dir / 'model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2, default=str)
    
    trainer.log_event(f"✅ Model information saved to: {trainer.results_dir / 'model_info.json'}")


if __name__ == '__main__':
    main()
