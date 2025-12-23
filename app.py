import os
import cv2
import numpy as np
from flask import Flask, render_template, request
import onnxruntime as ort

app = Flask(__name__)

# ===============================
# Load ONNX model
# ===============================
MODEL_PATH = "model.onnx"
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ===============================
# Class labels
# ===============================
class_names = [
    'Non X-ray',
    'XR_ELBOW',
    'XR_FINGER',
    'XR_FOREARM',
    'XR_HAND',
    'XR_HUMERUS',
    'XR_SHOULDER',
    'XR_WRIST'
]

CONFIDENCE_THRESHOLD = 95.0

# ===============================
# Upload config
# ===============================
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ===============================
# Routes
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None
    low_confidence = False

    if request.method == "POST":
        file = request.files.get("image")

        if file:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            # ---- Image preprocessing (must match training) ----
            image = cv2.imread(image_path)
            image = cv2.resize(image, (256, 256))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = image.astype(np.float32)
            image = image / 255.0                     # normalize
            image = np.expand_dims(image, axis=0)     # (1, 256, 256, 3)

            # ---- ONNX inference ----
            preds = session.run(
                [output_name],
                {input_name: image}
            )[0]  # shape: (1, num_classes)

            class_id = int(np.argmax(preds))
            confidence = float(preds[0][class_id] * 100)

            if confidence >= CONFIDENCE_THRESHOLD:
                prediction = class_names[class_id]
            else:
                low_confidence = True

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        low_confidence=low_confidence
    )


# ===============================
# Main
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
