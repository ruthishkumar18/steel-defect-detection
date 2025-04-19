import os
import uuid
import torch
from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
import cv2

# Add safe globals for PyTorch model loading
add_safe_globals([DetectionModel])  # Trust this model class

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load YOLO model
model = YOLO("best.pt")  # Replace with your trained model path

# Confidence threshold (adjustable)
CONFIDENCE_THRESHOLD = 0.5

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    # Save uploaded image
    filename = f"{uuid.uuid4().hex}.jpg"
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)

    try:
        # Run detection with confidence threshold
        results = model(input_path, conf=CONFIDENCE_THRESHOLD)
        result = results[0]

        # Check if any boxes are detected
        if not result.boxes or len(result.boxes.cls) == 0:
            return render_template("index.html", error="No steel surface defect detected or image is invalid.")

        # Save annotated image
        output_filename = f"pred_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        annotated_image = result.plot()
        cv2.imwrite(output_path, annotated_image)

        # Extract detected class labels and confidence
        defects = []
        for i in range(len(result.boxes.cls)):
            cls_id = int(result.boxes.cls[i])
            conf = float(result.boxes.conf[i])
            label = result.names[cls_id]
            defects.append({
                'label': label,
                'confidence': f"{conf * 100:.2f}%"
            })

        return render_template("index.html", result={
            'original': filename,
            'output': output_filename,
            'defects': defects
        })

    except Exception as e:
        print("Error during prediction:", str(e))
        return render_template("index.html", error="An error occurred during prediction. Please try again.")

if __name__ == '__main__':
    app.run(debug=True)
