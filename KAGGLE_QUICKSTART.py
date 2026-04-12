#!/usr/bin/env python3
"""
KAGGLE QUICK START GUIDE
خطوات سهلة لتشغيل التدريب المحسّن على Kaggle
"""

guide = """
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                  🚀 KAGGLE QUICK START (مع التحسينات)                       ║
║                                                                            ║
║              كيفية تشغيل التدريب المحسّن على Kaggle Notebooks               ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

خطوة 1️⃣: إذهب إلى Kaggle
═════════════════════════════════════════════════════════════════════════════

1. افتح: https://kaggle.com
2. سجّل الدخول (أو أنشئ حساب مجاني)
3. اذهب إلى: "Code" → "New Notebook"


خطوة 2️⃣: أضف مجلد البيانات الخاص بك
═════════════════════════════════════════════════════════════════════════════

1. في Kaggle Notebook، انقر على "Input" (أيقونة البيانات)
2. ابحث عن مجلد البيانات الذي حملته مسبقاً
   أو: أنشئ Dataset جديد وارفع:
   - human-footprint.zip
   - vehicle.zip
3. أضفه إلى الـ Notebook


خطوة 3️⃣: نسخ وألصق هذا الكود
═════════════════════════════════════════════════════════════════════════════

# الخلية 1️⃣: التثبيت
!pip install ultralytics -q
print("✅ Installation complete!")


# الخلية 2️⃣: تحضير البيانات
import os
import zipfile
from pathlib import Path

# استخرج البيانات
input_path = "/kaggle/input"
working_path = "/kaggle/working"

# ابحث عن الملفات المرفوعة
for item in os.listdir(input_path):
    item_path = os.path.join(input_path, item)
    if os.path.isdir(item_path):
        # ابحث عن ملفات zip
        for file in os.listdir(item_path):
            if file.endswith('.zip'):
                print(f"📦 Extracting {file}...")
                with zipfile.ZipFile(os.path.join(item_path, file), 'r') as zip_ref:
                    zip_ref.extractall(working_path)


# الخلية 3️⃣: تحقق من البيانات
import os
for folder in ['human-footprint', 'vehicle']:
    path = f'/kaggle/working/{folder}'
    if os.path.exists(path):
        train_count = len(os.listdir(f'{path}/train/images'))
        valid_count = len(os.listdir(f'{path}/valid/images'))
        test_count = len(os.listdir(f'{path}/test/images'))
        print(f"✅ {folder}:")
        print(f"   Train: {train_count} | Valid: {valid_count} | Test: {test_count}")


# الخلية 4️⃣: تشغيل التدريب المحسّن
from ultralytics import YOLO
import torch

print(f"🚀 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Human-Footprint Training
print("\\n🎓 Training Human-Footprint Model...")
model = YOLO('yolov8m.pt')  # Medium model
results_hf = model.train(
    data='/kaggle/working/human-footprint/data.yaml',
    epochs=100,        # Extended
    imgsz=1280,       # Higher resolution
    device=0,
    batch=8,
    patience=0,       # No early stopping
    verbose=True,
    augment=True,
)

# Vehicle Training
print("\\n🎓 Training Vehicle Model...")
model2 = YOLO('yolov8m.pt')
results_vehicle = model2.train(
    data='/kaggle/working/vehicle/data.yaml',
    epochs=100,        # Extended
    imgsz=1280,       # Higher resolution
    device=0,
    batch=8,
    patience=0,       # No early stopping
    verbose=True,
    augment=True,
)


# الخلية 5️⃣: التقييم والمقاييس
print("\\n📊 Evaluating Models...")

from pathlib import Path
import json

results_summary = {}

# Find trained models
for run_folder in Path('/kaggle/working/runs/detect').glob('train*'):
    model_path = run_folder / 'weights' / 'best.pt'
    if model_path.exists():
        print(f"\\n🔍 Evaluating {run_folder.name}...")
        model = YOLO(str(model_path))
        
        # Find corresponding dataset
        if 'footprint' in str(run_folder).lower() or '1' in str(run_folder):
            data_yaml = '/kaggle/working/human-footprint/data.yaml'
            dataset_name = 'human-footprint'
        else:
            data_yaml = '/kaggle/working/vehicle/data.yaml'
            dataset_name = 'vehicle'
        
        metrics = model.val(data=data_yaml)
        
        results_summary[dataset_name] = {
            'mAP50': float(metrics.box.map50),
            'mAP': float(metrics.box.map),
            'precision': float(metrics.box.p.mean()),
            'recall': float(metrics.box.r.mean()),
        }

# Print Results
print("\\n" + "="*80)
print("VALIDATION METRICS - التقييم")
print("="*80)
for dataset, metrics in results_summary.items():
    print(f"\\n{dataset}:")
    print(f"  mAP@50:   {metrics['mAP50']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")


# الخلية 6️⃣: الحفظ والعودة
print("\\n✅ Training complete!")
print("📁 Models saved in: /kaggle/working/runs/detect/")
print("📊 Check outputs in Kaggle notebook!")


═════════════════════════════════════════════════════════════════════════════


خطوة 4️⃣: شغّل الـ Notebook 🚀
═════════════════════════════════════════════════════════════════════════════

1. انقر "Run All" أو شغّل كل خلية تباعاً
2. اتركها تعمل (3-5 ساعات)
3. الـ Kaggle GPU سيعمل مجاناً! 🎉


خطوة 5️⃣: شاهد النتائج 📊
═════════════════════════════════════════════════════════════════════════════

بعد انتهاء التدريب:
- الخلية 5️⃣ ستطبع النتائج مباشرة
- مقاييس محسّنة جداً!
- يمكنك حفظ الـ Notebook كصورة (snapshot)
- شارك الـ Notebook مع فريقك!


النتائج المتوقعة ✨
═════════════════════════════════════════════════════════════════════════════

Human-Footprint:
  mAP@50:    0.50-0.65  (من 0.33)  ⬆️ +52-96%
  Recall:    0.60-0.75  (من 0.31)  ⬆️ +94-142%
  F1-Score:  0.60-0.70  (من 0.45)  ⬆️ +33-56%

Vehicle:
  mAP@50:    0.40-0.55  (من 0.17)  ⬆️ +135-224%
  Recall:    0.50-0.65  (من 0.19)  ⬆️ +163-242%
  F1-Score:  0.50-0.60  (من 0.26)  ⬆️ +92-131%


نصائح إضافية 💡
═════════════════════════════════════════════════════════════════════════════

✅ إذا كانت النتائج أفضل:
   - جرّب yolov8l بدل yolov8m للدقة الأعلى
   - زيادة epochs إلى 150-200

❌ إذا حصلت على OOM:
   - قلل batch من 8 إلى 4
   - قلل imgsz من 1280 إلى 768
   - استخدم yolov8n بدل yolov8m

⚠️ الـ Kaggle GPU:
   - مجاني 30 ساعة أسبوعياً
   - كافي لـ 3-4 تدريبات مثل هذا
   - توقف تلقائي بعد انقطاع 60 دقيقة


نموذج سريع - سكريبت واحد ⚡
═════════════════════════════════════════════════════════════════════════════

إذا أردت كل شيء في خلية واحدة على Kaggle:

```python
# Complete Training Pipeline
import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch

# Setup
os.chdir('/kaggle/working')
project_root = Path('/kaggle/working')

# Train
for dataset_name in ['human-footprint', 'vehicle']:
    print(f"\\n🚀 Training {dataset_name}...")
    model = YOLO('yolov8m.pt')
    
    results = model.train(
        data=str(project_root / dataset_name / 'data.yaml'),
        epochs=100,
        imgsz=1280,
        device=0,
        batch=8,
        patience=0,
        verbose=False,
    )

print("✅ Done!")
```


═══════════════════════════════════════════════════════════════════════════════

👉 الآن افتح Kaggle وابدأ! 🚀

Questions? Check:
- OPTIMIZATIONS_APPLIED.md (التحسينات)
- IMPROVEMENTS.txt (الفروقات)
- README_PIPELINE.md (التفاصيل الكاملة)

═══════════════════════════════════════════════════════════════════════════════
"""

if __name__ == '__main__':
    print(guide)
