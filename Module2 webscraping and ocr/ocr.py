from flask import Flask, request, render_template
import pytesseract
from PIL import Image
import cv2
import numpy as np
import io

app = Flask(__name__)

# Optional: Set Tesseract path if not in system PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

@app.route('/')
def home():
    return render_template('ocr.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    image = Image.open(file.stream)  # PIL image
    opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    text = pytesseract.image_to_string(opencvImage)
    return text

if __name__ == '__main__':
    app.run(debug=True)

