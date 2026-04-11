#!/usr/bin/env bash
# Setup verification script - Run this to verify everything is ready

echo "════════════════════════════════════════════════════════════════════════════"
echo "              OBJECT DETECTION TRAINING SETUP VERIFICATION"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# Check Python version
echo "1️⃣  Python version:"
python --version

# Check pip
echo ""
echo "2️⃣  Pip version:"
pip --version

# Check YOLOv8
echo ""
echo "3️⃣  YOLOv8 installation:"
python -c "from ultralytics import YOLO; print('   ✅ YOLOv8 installed')" 2>/dev/null || echo "   ⏳ Still installing..."

# Check other dependencies
echo ""
echo "4️⃣  Required packages:"
python -c "
import sys
packages = ['cv2', 'matplotlib', 'sklearn', 'pandas', 'numpy', 'PIL', 'yaml', 'tqdm']
missing = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f'   ✅ {pkg}')
    except:
        print(f'   ❌ {pkg} - installing...')
        missing.append(pkg)
" 2>/dev/null

# Check datasets
echo ""
echo "5️⃣  Dataset structure:"
echo "   Human-footprint:"
if [ -d "human-footprint/train" ]; then
    echo "      ✅ Train set exists"
fi
if [ -d "human-footprint/valid" ]; then
    echo "      ✅ Valid set exists"
fi
if [ -d "human-footprint/test" ]; then
    echo "      ✅ Test set exists"
fi

echo "   Vehicle:"
if [ -d "vehicle/train" ]; then
    echo "      ✅ Train set exists"
fi
if [ -d "vehicle/valid" ]; then
    echo "      ✅ Valid set exists"
fi
if [ -d "vehicle/test" ]; then
    echo "      ✅ Test set exists"
fi

# Check scripts
echo ""
echo "6️⃣  Training scripts:"
if [ -f "train_models.py" ]; then
    echo "   ✅ train_models.py"
fi
if [ -f "evaluate_models.py" ]; then
    echo "   ✅ evaluate_models.py"
fi
if [ -f "visualize_results.py" ]; then
    echo "   ✅ visualize_results.py"
fi
if [ -f "run_pipeline.py" ]; then
    echo "   ✅ run_pipeline.py"
fi

# Check GPU
echo ""
echo "7️⃣  GPU availability:"
python -c "
import torch
if torch.cuda.is_available():
    print(f'   ✅ GPU detected: {torch.cuda.get_device_name(0)}')
else:
    print('   ℹ️  GPU not available (CPU mode)')
"

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "                    SETUP VERIFICATION COMPLETE"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "🚀 NEXT STEP: Run python run_pipeline.py"
echo ""
