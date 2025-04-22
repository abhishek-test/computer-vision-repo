
from flask import Flask, request, render_template, redirect, url_for, Response
from werkzeug.utils import secure_filename
from sort import Sort 
import numpy as np
import torch
import cv2
import os

app = Flask(__name__, template_folder='../templates')
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}

# Load YOLOv5 model
model = torch.hub.load('C:\\Users\\24793/.cache\\torch\\hub\\ultralytics_yolov5_master', 
                       'custom', path='resources/yolov5s.pt', source='local') 

# Initialize SORT tracker
tracker = Sort()

@app.route('/video_feed')
def video_feed():
    filename = request.args.get('filename')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(generate_frames(filepath), mimetype='multipart/x-mixed-replace; boundary=frame')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    filename = request.args.get('filename')
    return render_template('index.html', filename=filename)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return redirect(url_for('index', filename=filename))
    return redirect(request.url)

@app.route('/reset', methods=['POST'])
def reset_tracker():
    global tracker
    tracker = Sort()  
    tracker.reset()
    return redirect(url_for('index'))

def generate_frames(filepath):
    cap = cv2.VideoCapture(filepath)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        detections = []
        for *xyxy, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            detections.append([x1, y1, x2, y2, conf.item()])
        
        # Update tracker with detections
        trackers = tracker.update(np.array(detections))
        
        # Draw bounding boxes and tracker IDs
        for d in trackers:
            x1, y1, x2, y2, track_id = int(d[0]), int(d[1]), int(d[2]), int(d[3]), int(d[4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/detect/<filename>')
def detect_objects(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(generate_frames(filepath), mimetype='multipart/x-mixed-replace; boundary=frame')
    

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if __name__ == '__main__':
    app.run(debug=True)