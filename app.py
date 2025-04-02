import base64
import json
import pytesseract
import cv2
import numpy as np
from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
from flask_cors import CORS  # Import CORS

app = Flask(__name__)

# Enable CORS for all domains (you can also restrict to specific domains)
CORS(app)

# Set Tesseract path for Windows users
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(image):
    """Convert image to grayscale, denoise, and apply thresholding."""
    img = np.array(image.convert('L'))  # Convert to grayscale
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Reduce noise
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Thresholding
    return Image.fromarray(img)

@app.route('/extract-json', methods=['POST'])
def extract_json():
    try:
        data = request.json
        image_data = data.get("imageBase64")

        if not image_data:
            return jsonify({"success": False, "message": "No image provided"}), 400

        # Decode base64 image
        encoded_str = image_data.split(",")[1] if "," in image_data else image_data
        image_bytes = base64.b64decode(encoded_str)
        image = Image.open(BytesIO(image_bytes))

        # Preprocess image
        processed_image = preprocess_image(image)

        # Perform OCR
        extracted_text = pytesseract.image_to_string(processed_image, config="--psm 6")

        # Debug: Print the extracted text
        print("\n--- Extracted Text ---\n", extracted_text, "\n--- End of Extracted Text ---\n")

        return jsonify({"success": True, "extracted_text": extracted_text, "message": "Raw OCR output"})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
