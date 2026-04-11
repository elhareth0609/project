#!/usr/bin/env python3
"""
Evaluation and Metrics Generation Script
Generates comprehensive validation metrics: mAP@50, Precision, Recall, F1
"""

import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


class MetricsGenerator:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.results_dir = self.project_root / 'training_results'
        self.metrics_dir = self.results_dir / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_metrics_from_model(self, model_path, dataset_name):
        """
        Evaluate model and extract validation metrics
        
        Returns:
            Dictionary with metrics: mAP@50, Precision, Recall, F1
        """
        print(f"\n{'='*60}")
        print(f"Evaluating: {dataset_name}")
        print(f"{'='*60}\n")
        
        try:
            # Load trained model
            model = YOLO(model_path)
            
            # Get data.yaml path
            if dataset_name == 'human-footprint':
                data_yaml = self.project_root / 'human-footprint' / 'data.yaml'
            else:
                data_yaml = self.project_root / dataset_name / 'data.yaml'
            
            if not data_yaml.exists():
                print(f"❌ data.yaml not found: {data_yaml}")
                return None
            
            # Validate on validation set
            print(f"📊 Running validation on: {data_yaml}")
            metrics = model.val(
                data=str(data_yaml),
                conf=0.25,
                iou=0.5,
                device=0
            )
            
            # Extract key metrics
            # Note: metrics object contains results for different IoU thresholds
            metrics_dict = {
                'dataset': dataset_name,
                'timestamp': datetime.now().isoformat(),
                'model_path': str(model_path),
                
                # mAP@50 (mean Average Precision at IoU 0.50)
                'mAP50': float(metrics.box.map50) if hasattr(metrics.box, 'map50') else None,
                
                # mAP@50:95 (mean Average Precision at IoU 0.50:0.95)
                'mAP': float(metrics.box.map) if hasattr(metrics.box, 'map') else None,
                
                # Per-class metrics
                'class_metrics': {}
            }
            
            # Extract per-class precision and recall if available
            if hasattr(metrics, 'box') and hasattr(metrics.box, 'p'):
                metrics_dict['precision'] = float(np.mean(metrics.box.p)) if len(metrics.box.p) > 0 else None
                metrics_dict['recall'] = float(np.mean(metrics.box.r)) if len(metrics.box.r) > 0 else None
                
                # Calculate F1 score
                if metrics_dict['precision'] and metrics_dict['recall']:
                    p = metrics_dict['precision']
                    r = metrics_dict['recall']
                    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
                    metrics_dict['f1_score'] = float(f1)
                
                # Per-class metrics
                if hasattr(metrics.box, 'p') and hasattr(metrics.box, 'r'):
                    for i, (precision, recall) in enumerate(zip(metrics.box.p, metrics.box.r)):
                        metrics_dict['class_metrics'][f'class_{i}'] = {
                            'precision': float(precision),
                            'recall': float(recall)
                        }
            
            return metrics_dict
            
        except Exception as e:
            print(f"❌ Error during evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_report(self, models_info):
        """Generate comprehensive metrics report"""
        print(f"\n{'='*60}")
        print("GENERATING METRICS REPORT")
        print(f"{'='*60}\n")
        
        all_metrics = []
        
        for dataset_name, model_info in models_info.items():
            if 'results_dir' in model_info:
                # Look for best.pt in the results directory
                results_dir = Path(model_info['results_dir'])
                best_model = results_dir / 'weights' / 'best.pt'
                
                if best_model.exists():
                    metrics = self.extract_metrics_from_model(str(best_model), dataset_name)
                    if metrics:
                        all_metrics.append(metrics)
                else:
                    print(f"❌ Best model not found: {best_model}")
        
        # Save detailed metrics
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            
            # Print summary table
            print("\n" + "="*80)
            print("VALIDATION METRICS SUMMARY")
            print("="*80)
            
            for metrics in all_metrics:
                print(f"\n📊 Dataset: {metrics['dataset']}")
                print(f"   mAP@50: {metrics.get('mAP50', 'N/A'):.4f}" if metrics.get('mAP50') else "   mAP@50: N/A")
                print(f"   mAP@50:95: {metrics.get('mAP', 'N/A'):.4f}" if metrics.get('mAP') else "   mAP@50:95: N/A")
                print(f"   Precision: {metrics.get('precision', 'N/A'):.4f}" if metrics.get('precision') else "   Precision: N/A")
                print(f"   Recall: {metrics.get('recall', 'N/A'):.4f}" if metrics.get('recall') else "   Recall: N/A")
                print(f"   F1-Score: {metrics.get('f1_score', 'N/A'):.4f}" if metrics.get('f1_score') else "   F1-Score: N/A")
            
            print("\n" + "="*80)
            
            # Save metrics as JSON
            metrics_file = self.metrics_dir / 'validation_metrics.json'
            with open(metrics_file, 'w') as f:
                json.dump(all_metrics, f, indent=2)
            print(f"\n✅ Metrics saved to: {metrics_file}")
            
            # Save metrics as CSV
            csv_file = self.metrics_dir / 'validation_metrics.csv'
            summary_df = pd.DataFrame([{
                'Dataset': m['dataset'],
                'mAP@50': f"{m.get('mAP50', 0):.4f}" if m.get('mAP50') else 'N/A',
                'mAP@50:95': f"{m.get('mAP', 0):.4f}" if m.get('mAP') else 'N/A',
                'Precision': f"{m.get('precision', 0):.4f}" if m.get('precision') else 'N/A',
                'Recall': f"{m.get('recall', 0):.4f}" if m.get('recall') else 'N/A',
                'F1-Score': f"{m.get('f1_score', 0):.4f}" if m.get('f1_score') else 'N/A'
            } for m in all_metrics])
            summary_df.to_csv(csv_file, index=False)
            print(f"✅ CSV report saved to: {csv_file}")
            
            return all_metrics
        else:
            print("❌ No metrics generated")
            return None


def main():
    project_root = Path("c:/Users/elhareth/Downloads/project")
    results_dir = project_root / 'training_results'
    
    # Load model info
    model_info_file = results_dir / 'model_info.json'
    if not model_info_file.exists():
        print(f"❌ Model info file not found: {model_info_file}")
        print("Please run train_models.py first")
        return
    
    with open(model_info_file, 'r') as f:
        model_info = json.load(f)
    
    generator = MetricsGenerator(project_root)
    metrics = generator.generate_report(model_info.get('trained_models', {}))
    
    if metrics:
        print(f"\n✅ Evaluation complete!")
    else:
        print(f"\n❌ Evaluation failed")


if __name__ == '__main__':
    main()
