# Final-Year Project — Campus Surveillance Dashboard

B.E. final-year project: a Flask dashboard for campus monitoring, combining a few computer-vision models behind a single UI and API.

## Features
- Gate management, people in/out, and social-distancing pages
- Mask detection on images, via a Keras CNN (`models/mask_keras_model.h5`)
- Emotion analysis on images and text, via the `fer` (Facial Expression Recognition) library
- A small JSON API (`/mask/detect/image/api/v1`, `/emotion/detect/image/api/v1`) alongside the browser pages, with `api-call-img.py` as a sample client

## Stack
Flask, TensorFlow/Keras, OpenCV, FER, Pillow.

## Use
1. `pip install flask tensorflow opencv-python fer pillow matplotlib` (no pinned requirements file was kept).
2. `python App/app.py`, then open `/dashboard`.

College project, 2019–2021.
