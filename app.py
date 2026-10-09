import os
import io
import base64

from flask import Flask, request, render_template_string
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# Maximum image upload size: 5 MB
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# Load the YOLO model
model = YOLO("yolo11n.pt")

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>YOLO Object Detection</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #141e30, #243b55);
            color: white;
            min-height: 100vh;
        }

        .container {
            width: 90%;
            max-width: 850px;
            margin: 40px auto;
            padding: 30px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        }

        h1 {
            color: #00e5ff;
            margin-bottom: 10px;
        }

        p {
            color: #dddddd;
            line-height: 1.6;
        }

        .upload-box {
            margin: 25px 0;
            padding: 25px;
            border: 2px dashed #00e5ff;
            border-radius: 12px;
        }

        input[type="file"] {
            display: block;
            margin: 15px auto;
            max-width: 100%;
        }

        button {
            background: #00e5ff;
            color: #141e30;
            border: none;
            padding: 12px 25px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
        }

        button:hover {
            background: #00bcd4;
        }

        .result {
            margin-top: 30px;
        }

        .result img {
            width: 100%;
            max-width: 650px;
            border-radius: 12px;
            margin-top: 15px;
        }

        .error {
            color: #ff8080;
            font-weight: bold;
        }

        .info {
            background: rgba(0, 229, 255, 0.1);
            padding: 12px;
            border-radius: 8px;
            margin-top: 20px;
        }

        footer {
            margin-top: 30px;
            color: #bbbbbb;
            font-size: 13px;
        }
    </style>
</head>

<body>
    <div class="container">

        <h1>YOLO Object Detection</h1>

        <p>
            Upload an image and let the YOLO deep learning model
            detect objects automatically.
        </p>

        <div class="upload-box">
            <form method="POST" enctype="multipart/form-data">
                <label for="image">
                    <strong>Select an image</strong>
                </label>

                <input
                    type="file"
                    id="image"
                    name="image"
                    accept=".jpg,.jpeg,.png,.webp"
                    required
                >

                <button type="submit">Detect Objects</button>
            </form>
        </div>

        {% if error %}
            <p class="error">{{ error }}</p>
        {% endif %}

        {% if result_image %}
            <div class="result">
                <h2>Detection Result</h2>

                <img
                    src="data:image/jpeg;base64,{{ result_image }}"
                    alt="Image with detected objects"
                >

                <div class="info">
                    <h3>Detected Objects</h3>
                    {% if detections %}
                        {% for detection in detections %}
                            <p>
                                {{ detection.name }}
                                — Confidence: {{ detection.confidence }}%
                            </p>
                        {% endfor %}
                    {% else %}
                        <p>No objects were detected.</p>
                    {% endif %}
                </div>
            </div>
        {% endif %}

        <footer>
            Powered by YOLO and Flask
        </footer>

    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    result_image = None
    detections = []
    error = None

    if request.method == "POST":
        if "image" not in request.files:
            error = "Please select an image."
            return render_template_string(
                HTML_PAGE,
                result_image=result_image,
                detections=detections,
                error=error
            )

        file = request.files["image"]

        if file.filename == "":
            error = "Please select an image."
        else:
            try:
                # Read the uploaded image
                image = Image.open(file.stream).convert("RGB")

                # Run YOLO object detection
                results = model.predict(
                    source=image,
                    conf=0.25,
                    verbose=False
                )

                # Draw detection boxes on the image
                annotated_image = results[0].plot()

                # Convert the result to JPEG format
                output_image = Image.fromarray(annotated_image[:, :, ::-1])

                buffer = io.BytesIO()
                output_image.save(buffer, format="JPEG")

                # Convert the image to Base64 for displaying on the webpage
                result_image = base64.b64encode(
                    buffer.getvalue()
                ).decode("utf-8")

                # Collect detected object names and confidence scores
                for box in results[0].boxes:
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())

                    detections.append({
                        "name": model.names[class_id],
                        "confidence": round(confidence * 100, 2)
                    })

            except Exception as e:
                app.logger.exception("Image detection failed")
                error = "Could not process the image. Please upload a valid image."

    return render_template_string(
        HTML_PAGE,
        result_image=result_image,
        detections=detections,
        error=error
    )


@app.errorhandler(413)
def file_too_large(error):
    return "Image too large. Please upload an image smaller than 5 MB.", 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

