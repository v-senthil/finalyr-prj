from flask import Flask,render_template, url_for, request, make_response, Response, redirect, session, jsonify
from werkzeug.utils import secure_filename
import os
from helpers import image2base64
import requests
import json
from fer import FER
import matplotlib.pyplot as plt
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2

app = Flask(__name__)
app.secret_key = os.urandom(24)

#Upload Folder
UPLOAD_FOLDER = 'static\\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Home Page
@app.route("/")
def index():
    return render_template("index.html")

#Dashboard
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# Gate Managment
@app.route("/gatemanage")
def gatemanage():
    return render_template("gatemanage.html")

# Mask Detection Video
@app.route("/maskdetection")
def maskdetection():
    return render_template("maskdetectionvideo.html")

# People in and out
@app.route("/peopleinout")
def peopleinout():
    return render_template("peopleinout.html")

# Emotion Analysis Video
@app.route("/emotionanalysis")
def emotionanalysis():
    return render_template("emotionanalysisvideo.html")

# Emotion Analysis Text
@app.route("/emotionanalysistext")
def emotionanalysistext():
    return render_template("emotionanalysistext.html")

# Social Distancing
@app.route("/socialdistance")
def socialdistance():
    return render_template("socialdistance.html")

# Documentation
@app.route("/docs")
def docs():
    return render_template("docs.html")

# API
@app.route("/api")
def api():
    return render_template("api.html")


#Mask Detection Image
@app.route("/mask/detect/image", methods=["GET", "POST"])
def maskdetectionimage():
    if request.method == 'POST':
        f = request.files['image']
        if 'png' in f.filename:
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename("maskdetectionimage.png")))

            url = 'http://localhost:38100/predict/6ecec6fe-5a50-4bf2-8416-c4c5cc0a75cc'
            x = image2base64(r'static\uploads\maskdetectionimage.png')
            data = {'inputs': {'Image': x}}
            r = requests.post(url, data=json.dumps(data))
            success = json.loads(r.text)['outputs']['Prediction']
            return render_template("maskdetectimage.html", success = success)
        else:
            success = "Invalid format, only PNG is allowed"
            return render_template("maskdetectimage.html", success=success)
    else:
        return render_template("maskdetectimage.html")

#Emotion Analysis Image
@app.route("/emotion/detect/image", methods=["GET", "POST"])
def emotionanalysisimage():
    if request.method == 'POST':
        f = request.files['image']
        if 'jpg' in f.filename:
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename("emotion-detect-image.jpg")))

            img = plt.imread("static\\uploads\\emotion-detect-image.jpg")
            detector = FER(mtcnn=True)
            emotion, score = detector.top_emotion(img)

            return render_template("emotiondetectimage.html", success = emotion)
        else:
            success = "Invalid format, only JPG is allowed"
            return render_template("emotiondetectimage.html", success=success)
    else:
        return render_template("emotiondetectimage.html")






###################  API Calls ###################

# API call for emotion image
@app.route("/emotion/detect/image/api/v1", methods=["POST"])
def emotiondetectimageapi():
    f = request.files['image']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename("emotion-detect-image-api.jpg")))
    request.headers.get("X-Requested-With") == "XMLHttpRequest"
    img = plt.imread(r"static\uploads\emotion-detect-image-api.jpg")
    detector = FER(mtcnn=True)
    emotion, score = detector.top_emotion(img)
    return jsonify({'msg': 'success', 'data': detector.detect_emotions(img), 'emotion': emotion, 'score': score})

# API call for mask image
@app.route("/mask/detect/image/api/v1", methods=["POST"])
def maskdetectimageapi():
    f = request.files['image']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename("maskdetectionimage-api.jpg")))
    request.headers.get("X-Requested-With") == "XMLHttpRequest"
    url = 'http://localhost:38100/predict/6ecec6fe-5a50-4bf2-8416-c4c5cc0a75cc'
    x = image2base64(r'static\uploads\maskdetectionimage-api.jpg')
    data = {'inputs': {'Image': x}}
    r = requests.post(url, data=json.dumps(data))
    labels = json.loads(r.text)['outputs']
    success = labels['Prediction']
    return jsonify({'msg': 'success', 'Prediction': success, 'data': labels})

# API call from emotion text

if __name__ == "__main__":
    app.run(debug=True)
