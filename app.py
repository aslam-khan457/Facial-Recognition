import os
import base64
from flask import Flask, render_template, request, jsonify, redirect
import cv2
import numpy as np
from utils import detect_emotions

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    if file:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        result_frame, emotion_label = detect_emotions(img)

        _, buffer = cv2.imencode('.jpg', result_frame)
        img_bytes = buffer.tobytes()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')

        return render_template('index.html', result_img=f'data:image/jpeg;base64,{img_b64}', emotion=emotion_label or "No face detected")

    return redirect('/')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image data'}), 400

    try:
        # Decode base64 image from browser
        img_data = base64.b64decode(data['image'].split(',')[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Detect emotions
        result_frame, emotion_label = detect_emotions(frame)

        # Encode result frame back to base64
        _, buffer = cv2.imencode('.jpg', result_frame)
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'image': f'data:image/jpeg;base64,{img_b64}',
            'emotion': emotion_label or "No face detected"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
