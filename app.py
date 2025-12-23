import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# Load model
model = load_model("Fracture_Detection_Model.h5")

# Class labels
class_names = ['Non X-ray', 'XR_ELBOW', 'XR_FINGER', 'XR_FOREARM',
            'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST'
            ]

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
CONFIDENCE_THRESHOLD = 95.0


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None
    low_confidence = False

    if request.method == "POST":
        file = request.files["image"]

        if file:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            image = cv2.imread(image_path)
            image = cv2.resize(image, (256, 256))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            img_array = img_to_array(image)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array, verbose=0)
            class_id = np.argmax(preds)
            confidence = preds[0][class_id] * 100

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



if __name__ == "__main__":
    app.run(debug=True)
