import asyncio
import gradio as gr
import cv2
import numpy as np
import torch
import time
import redis.asyncio as redis
import sqlite3
from datetime import datetime

# Initialize Redis
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)

# Initialize SQLite
conn = sqlite3.connect('video_data.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS frame_data (
                frame_id INTEGER,
                timestamp TEXT,
                object_count INTEGER,
                capture_fps REAL,
                process_fps REAL,
                display_fps REAL
            )''')
conn.commit()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Global state
frame_counter = 0
reset_event = asyncio.Event()

async def capture_frames(video_path):
    global frame_counter
    cap = cv2.VideoCapture(video_path)
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        current_time = time.time()
        capture_fps = 1 / (current_time - prev_time)
        prev_time = current_time

        _, buffer = cv2.imencode('.jpg', frame)
        await redis_client.xadd('capture_stream', {'frame': buffer.tobytes(), 'frame_id': frame_counter, 'capture_fps': capture_fps})

        if reset_event.is_set():
            break

    cap.release()

async def process_frames():
    last_id = '0-0'
    prev_time = time.time()

    while not reset_event.is_set():
        response = await redis_client.xread({'capture_stream': last_id}, block=1000, count=1)
        if response:
            stream, messages = response[0]
            for msg_id, msg_data in messages:
                last_id = msg_id
                frame = np.frombuffer(msg_data[b'frame'], dtype=np.uint8)
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                frame_id = int(msg_data[b'frame_id'])
                capture_fps = float(msg_data[b'capture_fps'])

                current_time = time.time()
                process_fps = 1 / (current_time - prev_time)
                prev_time = current_time

                results = model(frame)
                object_count = len(results.xyxy[0])

                annotated_frame = results.render()[0]
                _, buffer = cv2.imencode('.jpg', annotated_frame)

                await redis_client.xadd('process_stream', {
                    'frame': buffer.tobytes(),
                    'frame_id': frame_id,
                    'object_count': object_count,
                    'capture_fps': capture_fps,
                    'process_fps': process_fps
                })

async def display_frames():
    last_id = '0-0'
    prev_time = time.time()

    while not reset_event.is_set():
        response = await redis_client.xread({'process_stream': last_id}, block=1000, count=1)
        if response:
            stream, messages = response[0]
            for msg_id, msg_data in messages:
                last_id = msg_id
                frame = np.frombuffer(msg_data[b'frame'], dtype=np.uint8)
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                frame_id = int(msg_data[b'frame_id'])
                object_count = int(msg_data[b'object_count'])
                capture_fps = float(msg_data[b'capture_fps'])
                process_fps = float(msg_data[b'process_fps'])

                current_time = time.time()
                display_fps = 1 / (current_time - prev_time)
                prev_time = current_time

                # Save to SQLite
                timestamp = datetime.now().isoformat()
                c.execute("INSERT INTO frame_data VALUES (?, ?, ?, ?, ?, ?)",
                          (frame_id, timestamp, object_count, capture_fps, process_fps, display_fps))
                conn.commit()

                # Overlay FPS info
                info_text = f'Frame: {frame_id} | Objects: {object_count} | Capture FPS: {capture_fps:.2f} | Process FPS: {process_fps:.2f} | Display FPS: {display_fps:.2f}'
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                _, buffer = cv2.imencode('.jpg', frame)
                frame_b64 = buffer.tobytes()

                yield frame_b64

async def reset_system():
    reset_event.set()
    await asyncio.sleep(1)
    await redis_client.delete('capture_stream')
    await redis_client.delete('process_stream')
    reset_event.clear()

async def main(video_path):
    await reset_system()
    capture_task = asyncio.create_task(capture_frames(video_path))
    process_task = asyncio.create_task(process_frames())
    await asyncio.sleep(1)  # Allow queues to fill
    return display_frames(), capture_task, process_task

def start_system(video):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    display_stream, capture_task, process_task = loop.run_until_complete(main(video))
    return gr.Stream(display_stream), "Processing Started!"

def reset():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(reset_system())
    return None, "System Reset!"

with gr.Blocks() as app:
    gr.Markdown("## Asynchronous Video Processing with YOLOv5, Redis, SQLite, and FPS Tracking 🚀")

    with gr.Row():
        video_input = gr.Video(label="Upload Video")
        start_button = gr.Button("Start Processing")
        reset_button = gr.Button("Reset")

    output_video = gr.Image(label="Processed Video Stream")
    status = gr.Textbox(label="Status")

    start_button.click(fn=start_system, inputs=video_input, outputs=[output_video, status])
    reset_button.click(fn=reset, inputs=None, outputs=[output_video, status])

app.launch()