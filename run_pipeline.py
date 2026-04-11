#!/usr/bin/env python3
"""
Master Pipeline Runner
Orchestrates entire workflow: data splitting, training, evaluation, visualization
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json


class PipelineOrchestrator:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.log_file = self.project_root / 'pipeline_execution.log'
        
    def log(self, message):
        """Log message to console and file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def run_script(self, script_name, description):
        """Run a Python script and capture output"""
        script_path = self.project_root / script_name
        
        if not script_path.exists():
            self.log(f"❌ Script not found: {script_path}")
            return False
        
        self.log(f"\n{'='*70}")
        self.log(f"STEP: {description}")
        self.log(f"Script: {script_name}")
        self.log(f"{'='*70}\n")
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=False,
                text=True,
                cwd=str(self.project_root)
            )
            
            if result.returncode == 0:
                self.log(f"✅ {description} - COMPLETED\n")
                return True
            else:
                self.log(f"❌ {description} - FAILED\n")
                return False
                
        except Exception as e:
            self.log(f"❌ Error running {script_name}: {str(e)}\n")
            return False
    
    def run_pipeline(self):
        """Execute complete pipeline"""
        self.log("\n" + "="*70)
        self.log(" "*10 + "OBJECT DETECTION MODEL TRAINING PIPELINE")
        self.log("="*70 + "\n")
        
        self.log(f"Project Root: {self.project_root}")
        self.log(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        steps = [
            ('train_models.py', '1. Training Models (Human-Footprint & Vehicle)'),
            ('evaluate_models.py', '2. Evaluating Models & Computing Metrics'),
            ('visualize_results.py', '3. Generating Visualizations & Reports'),
        ]
        
        results = {}
        
        for script, description in steps:
            success = self.run_script(script, description)
            results[script] = success
            
            if not success and description.startswith('1'):
                # If training fails, stop pipeline
                self.log("\n⚠️ Training failed, cannot proceed with evaluation")
                break
        
        # Summary
        self.log("\n" + "="*70)
        self.log("PIPELINE EXECUTION SUMMARY")
        self.log("="*70 + "\n")
        
        for script, success in results.items():
            status = "✅ COMPLETED" if success else "❌ FAILED"
            self.log(f"{script}: {status}")
        
        self.log(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("\n" + "="*70)
        
        # Check if all steps completed
        if all(results.values()):
            self.log("\n🎉 PIPELINE EXECUTION SUCCESSFUL!")
            self.log("\n📊 Results available in: training_results/")
            self.log("   ├── models/              (trained models)")
            self.log("   ├── metrics/             (validation metrics)")
            self.log("   ├── visualizations/      (charts and graphs)")
            self.log("   └── training_log.txt     (detailed log)")
            return True
        else:
            self.log("\n❌ PIPELINE EXECUTION FAILED!")
            return False


def main():
    project_root = Path("c:/Users/elhareth/Downloads/project")
    orchestrator = PipelineOrchestrator(project_root)
    
    success = orchestrator.run_pipeline()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
