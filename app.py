from flask import Flask,request,jsonify
from ultralytics import YOLO
import numpy as np
import os
from PIL import Image

app = Flask(__name__)
model = YOLO(r'C:\Users\asyon\OneDrive\Desktop\Coding\YOLO\runs\classify\train10\weights\last.pt')
def home():
    return "Welcome to the Flask App"

@app.route('/predict', methods = ['POST'])

def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file found'}), 404

    
    file = request.files['file']
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    results = model(file_path)
    os.remove(file_path)
    names_dict = results[0].names
    probs = results[0].probs.tolist()
    prediction = names_dict[np.argmax(probs)]

    prediction = 'Healthy'

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok = True)
    app.run(host = '0.0.0.0', port = 5000)