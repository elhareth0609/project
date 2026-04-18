#!/usr/bin/env python3
"""
Use Trained YOLO11 Model for Footprint Detection
"""

from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from pathlib import Path

def load_model(model_path='best.pt'):
    """Load trained YOLO model"""
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    return model

def predict_single_image(model, image_path, conf_threshold=0.25):
    """
    Predict on a single image
    
    Args:
        model: YOLO model
        image_path: Path to image
        conf_threshold: Confidence threshold (0-1)
    
    Returns:
        results: YOLO detection results
    """
    print(f"\n🔍 Predicting on: {image_path}")
    
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=False,
        verbose=False
    )
    
    # Parse results
    for result in results:
        print(f"\n📊 Detections found: {len(result.boxes)}")
        
        for i, box in enumerate(result.boxes):
            class_id = int(box.cls)
            class_name = result.names[class_id]
            confidence = box.conf.item()
            
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            print(f"\n  Detection {i+1}:")
            print(f"    Class: {class_name}")
            print(f"    Confidence: {confidence:.2%}")
            print(f"    Box: ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})")
    
    return results

def predict_batch(model, images_dir, conf_threshold=0.25, save_results=True):
    """
    Predict on multiple images
    
    Args:
        model: YOLO model
        images_dir: Directory containing images
        conf_threshold: Confidence threshold
        save_results: Save annotated images
    """
    images_dir = Path(images_dir)
    images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    
    print(f"\n🔍 Predicting on {len(images)} images from: {images_dir}")
    
    results = model.predict(
        source=images,
        conf=conf_threshold,
        save=save_results,
        project='predictions',
        name='results'
    )
    
    # Count detections
    total_detections = sum(len(r.boxes) for r in results)
    print(f"\n✅ Total detections: {total_detections}")
    
    if save_results:
        print(f"   Annotated images saved to: predictions/results/")
    
    return results

def draw_results(image_path, result, output_path=None):
    """
    Draw detection boxes on image and display/save
    
    Args:
        image_path: Path to original image
        result: YOLO result object
        output_path: Where to save annotated image
    """
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    
    # Draw each detection
    for box in result.boxes:
        # Get coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        
        # Get class and confidence
        class_id = int(box.cls)
        class_name = result.names[class_id]
        confidence = box.conf.item()
        
        # Colors for different classes
        colors = {
            'animal_footprint': (0, 255, 0),      # Green
            'human_footprint': (255, 0, 0)        # Blue
        }
        color = colors.get(class_name, (0, 0, 255))
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2%}"
        cv2.putText(
            img, label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
    
    if output_path:
        cv2.imwrite(str(output_path), img)
        print(f"✅ Annotated image saved: {output_path}")
    
    return img

def count_detections_by_class(results):
    """Count detections by class"""
    counts = {}
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = result.names[class_id]
            counts[class_name] = counts.get(class_name, 0) + 1
    
    print("\n📊 Detection Summary:")
    for class_name, count in counts.items():
        print(f"   {class_name}: {count} detections")
    
    return counts

def export_results_csv(results, output_file='predictions.csv'):
    """Export predictions to CSV"""
    import csv
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Class', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])
        
        for result in results:
            image_name = Path(result.path).name
            
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = result.names[class_id]
                confidence = box.conf.item()
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                writer.writerow([
                    image_name,
                    class_name,
                    f"{confidence:.4f}",
                    f"{x1:.1f}",
                    f"{y1:.1f}",
                    f"{x2:.1f}",
                    f"{y2:.1f}"
                ])
    
    print(f"✅ Results exported to: {output_file}")

# ============================================================
# USAGE EXAMPLES
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("YOLO11 FOOTPRINT DETECTION - INFERENCE")
    print("="*60)
    
    # Example 1: Load model
    # model = load_model('path/to/best.pt')
    
    # Example 2: Predict on single image
    # results = predict_single_image(model, 'footprint.jpg', conf_threshold=0.25)
    
    # Example 3: Predict on folder
    # results = predict_batch(model, './test_images', conf_threshold=0.25)
    
    # Example 4: Draw results
    # if results:
    #     draw_results('footprint.jpg', results[0], 'output.jpg')
    
    # Example 5: Export results
    # export_results_csv(results, 'predictions.csv')
    
    # Example 6: Count by class
    # count_detections_by_class(results)
    
    print("\n📝 QUICK START CODE:")
    print("""
from ultralytics import YOLO

# Load model
model = YOLO('best.pt')

# Predict
results = model.predict(source='image.jpg', conf=0.25)

# Get results
for result in results:
    for box in result.boxes:
        print(f"Class: {result.names[int(box.cls)]}")
        print(f"Confidence: {box.conf:.2%}")
        print(f"Box: {box.xyxy}")
    """)
    
    print("="*60)
