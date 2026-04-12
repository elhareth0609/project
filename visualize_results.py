#!/usr/bin/env python3
"""
Visualization and Results Dashboard
Creates visualizations for training results and metrics
"""

import json
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

matplotlib.use('Agg')  # Use non-interactive backend


class ResultsVisualizer:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.results_dir = self.project_root / 'training_results'
        self.viz_dir = self.results_dir / 'visualizations'
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
    def create_metrics_visualization(self):
        """Create visualizations for validation metrics"""
        metrics_file = self.results_dir / 'metrics' / 'validation_metrics.json'
        
        if not metrics_file.exists():
            print(f"❌ Metrics file not found: {metrics_file}")
            return
        
        print("\n📊 Creating visualizations...")
        
        with open(metrics_file, 'r') as f:
            metrics_list = json.load(f)
        
        # Create metrics DataFrame
        metrics_data = []
        for m in metrics_list:
            metrics_data.append({
                'Dataset': m['dataset'],
                'mAP@50': m.get('mAP50', 0),
                'mAP@50:95': m.get('mAP', 0),
                'Precision': m.get('precision', 0),
                'Recall': m.get('recall', 0),
                'F1-Score': m.get('f1_score', 0)
            })
        
        df = pd.DataFrame(metrics_data)
        
        # 1. Metrics Comparison Bar Chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics_cols = ['mAP@50', 'mAP@50:95', 'Precision', 'Recall', 'F1-Score']
        x = range(len(df))
        width = 0.15
        
        for i, metric in enumerate(metrics_cols):
            offset = (i - 2) * width
            ax.bar([xi + offset for xi in x], df[metric], width, label=metric)
        
        ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Validation Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Dataset'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        chart_path = self.viz_dir / '01_metrics_comparison.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {chart_path}")
        plt.close()
        
        # 2. Per-Dataset Metrics Radar Chart
        for idx, row in df.iterrows():
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            
            categories = ['mAP@50', 'Precision', 'Recall', 'F1-Score']
            values = [
                row['mAP@50'],
                row['Precision'],
                row['Recall'],
                row['F1-Score']
            ]
            
            # Complete the circle
            values += values[:1]
            angles = [n / float(len(categories)) * 2 * 3.14159 for n in range(len(categories))]
            angles += angles[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label='Score')
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1.0)
            ax.set_title(f"{row['Dataset']} - Performance Radar", 
                        fontsize=12, fontweight='bold', pad=20)
            ax.grid(True)
            
            plt.tight_layout()
            radar_path = self.viz_dir / f'02_{row["Dataset"]}_radar.png'
            plt.savefig(radar_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {radar_path}")
            plt.close()
        
        # 3. Metrics Heatmap
        fig, ax = plt.subplots(figsize=(8, 4))
        
        heatmap_data = df[['mAP@50', 'mAP@50:95', 'Precision', 'Recall', 'F1-Score']].values
        im = ax.imshow(heatmap_data, cmap='YlGn', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(range(len(metrics_cols)))
        ax.set_yticks(range(len(df)))
        ax.set_xticklabels(metrics_cols, rotation=45, ha='right')
        ax.set_yticklabels(df['Dataset'])
        
        # Add text annotations
        for i in range(len(df)):
            for j in range(len(metrics_cols)):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=10, fontweight='bold')
        
        ax.set_title('Metrics Heatmap (Validation Set)', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Score')
        plt.tight_layout()
        
        heatmap_path = self.viz_dir / '03_metrics_heatmap.png'
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {heatmap_path}")
        plt.close()
        
        print(f"\n✅ All visualizations saved to: {self.viz_dir}")
    
    def create_summary_report(self):
        """Create a text summary report"""
        metrics_file = self.results_dir / 'metrics' / 'validation_metrics.json'
        
        if not metrics_file.exists():
            return
        
        with open(metrics_file, 'r') as f:
            metrics_list = json.load(f)
        
        report_path = self.viz_dir / 'METRICS_REPORT.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(" "*15 + "OBJECT DETECTION MODEL VALIDATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Project Root: {self.project_root}\n\n")
            
            f.write("VALIDATION METRICS (Valid Set)\n")
            f.write("-"*70 + "\n\n")
            
            for metrics in metrics_list:
                f.write(f"Dataset: {metrics['dataset']}\n")
                f.write(f"{'─'*70}\n")
                f.write(f"  mAP@50 (Mean Average Precision at IoU=0.50):\n")
                f.write(f"    └─ {metrics.get('mAP50', 'N/A'):.4f}\n\n")
                f.write(f"  mAP@50:95 (Mean Average Precision across IoU thresholds):\n")
                f.write(f"    └─ {metrics.get('mAP', 'N/A'):.4f}\n\n")
                f.write(f"  Precision (Positive Predictive Value):\n")
                f.write(f"    └─ {metrics.get('precision', 'N/A'):.4f}\n\n")
                f.write(f"  Recall (True Positive Rate):\n")
                f.write(f"    └─ {metrics.get('recall', 'N/A'):.4f}\n\n")
                f.write(f"  F1-Score (Harmonic Mean of Precision and Recall):\n")
                f.write(f"    └─ {metrics.get('f1_score', 'N/A'):.4f}\n\n")
                f.write(f"{'─'*70}\n\n")
            
            f.write("\nMETRICS EXPLANATION\n")
            f.write("-"*70 + "\n")
            f.write("""
mAP@50:
  - Average Precision at 50% Intersection over Union threshold
  - Higher values indicate better model performance
  - Target: > 0.5-0.7 for good detection

Precision:
  - Of all detected objects, how many were correct?
  - True Positives / (True Positives + False Positives)
  - Higher is better, but needs balance with Recall

Recall:
  - Of all actual objects, how many did we find?
  - True Positives / (True Positives + False Negatives)
  - Higher is better, but needs balance with Precision

F1-Score:
  - Harmonic mean of Precision and Recall
  - Balanced metric when precision/recall tradeoff matters
  - Range: 0-1, higher is better

Target Metrics:
  - mAP@50: > 0.5
  - Precision: > 0.7
  - Recall: > 0.6
  - F1-Score: > 0.65
""")
            f.write("-"*70 + "\n")
        
        print(f"✅ Report saved: {report_path}")


def main():
    project_root = Path("/kaggle/working/project")
    visualizer = ResultsVisualizer(project_root)
    
    visualizer.create_metrics_visualization()
    visualizer.create_summary_report()
    
    print(f"\n✅ Visualization complete!")


if __name__ == '__main__':
    main()
