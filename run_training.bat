@echo off
REM Batch file for Windows - Run the complete pipeline

echo.
echo ========================================================================
echo.
echo        YOLOV8 OBJECT DETECTION MODEL TRAINING PIPELINE
echo.
echo        From labeled data to production-ready detection models
echo        with Validation Metrics (mAP@50, Precision, Recall, F1)
echo.
echo ========================================================================
echo.

cd /d "%~dp0"

echo Checking installation status...
echo.

REM Check if pip installation was successful
python -c "from ultralytics import YOLO" >nul 2>&1
if errorlevel 1 (
    echo ⏳ Still waiting for YOLOv8 installation to complete...
    echo    This happens automatically in background.
    echo    Waiting 30 seconds before retry...
    timeout /t 30 /nobreak
    python -c "from ultralytics import YOLO" >nul 2>&1
    if errorlevel 1 (
        echo ❌ YOLOv8 not yet installed. Installation in progress.
        echo    Please wait and run this script again in a few minutes.
        pause
        exit /b 1
    )
)

echo ✅ YOLOv8 and dependencies ready!
echo.
echo ========================================================================
echo RUNNING COMPLETE TRAINING PIPELINE
echo ========================================================================
echo.
echo This will:
echo   1. Train human-footprint detection model
echo   2. Train vehicle detection model
echo   3. Evaluate both models on validation set
echo   4. Compute metrics: mAP@50, Precision, Recall, F1
echo   5. Generate visualizations and reports
echo.
echo Estimated time: 2-4 hours with GPU (4-6 hours with CPU)
echo.
echo All results will be saved to: training_results\
echo.
echo ========================================================================
echo.

python run_pipeline.py

if errorlevel 1 (
    echo.
    echo ❌ Pipeline encountered an error.
    echo See pipeline_execution.log for details.
    pause
    exit /b 1
)

echo.
echo ✅ PIPELINE COMPLETED SUCCESSFULLY!
echo.
echo 📊 Results available in:
echo    - training_results\metrics\validation_metrics.json
echo    - training_results\visualizations\
echo    - training_results\models\
echo.
echo 🎉 Your models are ready for inference!
echo.
pause
