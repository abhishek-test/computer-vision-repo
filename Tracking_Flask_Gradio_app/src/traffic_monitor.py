import asyncio
import gradio as gr
import cv2
import torch
import numpy as np
import sqlite3
import json
import os
import time
import redis.asyncio as redis
from datetime import datetime
from fastapi import FastAPI
import uvicorn
import logging
from typing import List, Dict, Any
import threading
from queue import Queue

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Traffic-Monitor")

# Redis Setup
REDIS_URL = "redis://127.0.0.1:6379"
STREAM_PREFIX = "traffic"
GROUP_NAME = "traffic_group"
CONSUMER_NAME = f"consumer-{int(time.time())}"

# Database Setup
DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "traffic_data.db")

# FPS Monitor Class
class FPSMonitor:
    def __init__(self, name: str):
        self.name = name
        self.start_time = time.time()
        self.frames = 0
        self.last_update = time.time()
        self.current_fps = 0
        self.lock = threading.Lock()

    def update(self):
        with self.lock:
            self.frames += 1
            current_time = time.time()
            elapsed = current_time - self.last_update
            
            if elapsed >= 1.0:
                self.current_fps = self.frames / elapsed
                self.frames = 0
                self.last_update = current_time

    def get_fps(self) -> float:
        with self.lock:
            return round(self.current_fps, 2)

# Initialize FPS Monitors
capture_monitor = FPSMonitor("Capture")
processing_monitor = FPSMonitor("Processing")
logging_monitor = FPSMonitor("Logging")

# Database Functions
def ensure_db_dir():
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
        logger.info(f"Created database directory: {DB_DIR}")

def get_db_connection():
    ensure_db_dir()
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS vehicle_counts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        vehicle_count INTEGER,
        frame_id TEXT,
        video_source TEXT,
        fps_capture REAL,
        fps_processing REAL,
        fps_logging REAL
    )
    ''')
    
    cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_timestamp 
    ON vehicle_counts(timestamp)
    ''')
    
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at: {DB_PATH}")

def store_vehicle_count(vehicle_count: int, frame_id: str, video_source: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """INSERT INTO vehicle_counts 
            (vehicle_count, frame_id, video_source, fps_capture, fps_processing, fps_logging) 
            VALUES (?, ?, ?, ?, ?, ?)""",
            (vehicle_count, frame_id, video_source, 
             capture_monitor.get_fps(), 
             processing_monitor.get_fps(), 
             logging_monitor.get_fps())
        )
        conn.commit()
        logger.info(f"Stored vehicle count {vehicle_count} for frame {frame_id}")
    except Exception as e:
        logger.error(f"Error storing vehicle count: {e}")
        raise
    finally:
        conn.close()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.eval()
model.amp = True

# Define vehicle classes
VEHICLE_CLASSES = [2, 5, 7, 3]  # COCO class IDs for vehicles

# Global variables
redis_client = None
frame_queue = Queue(maxsize=30)
processed_frame_queue = Queue(maxsize=30)
shutdown_event = threading.Event()
loop = None

# Video Processing Functions
def process_frame(frame: np.ndarray) -> tuple:
    """Process a single frame with YOLOv5 and return annotated frame and detections"""
    processing_monitor.update()
    
    # Move model to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Convert frame to tensor and move to device
    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().to(device) / 255.0
    
    # Ensure tensor is divisible by 32 (YOLOv5 requirement)
    h, w = frame_tensor.shape[1:]
    h = (h // 32) * 32
    w = (w // 32) * 32
    frame_tensor = frame_tensor[:, :h, :w]
    
    # Add batch dimension and ensure correct shape
    frame_tensor = frame_tensor.unsqueeze(0)
    
    # Perform inference with autocast
    with torch.amp.autocast(device_type=device, enabled=model.amp):
        results = model(frame_tensor)
    
    # Process detections
    detections = []
    for det in results[0]:  # Get detections for first (and only) image in batch
        x1, y1, x2, y2, conf, cls = det
        if int(cls) in VEHICLE_CLASSES and conf > 0.5:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(conf),
                'class': int(cls),
                'class_name': model.names[int(cls)]
            })
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Add FPS counters to frame
    fps_text = f"Capture: {capture_monitor.get_fps():.1f} | "
    fps_text += f"Process: {processing_monitor.get_fps():.1f} | "
    fps_text += f"Log: {logging_monitor.get_fps():.1f}"
    cv2.putText(frame, fps_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame, detections

async def process_video_interface(video_path: str):
    """Gradio interface for video processing"""
    if not os.path.exists(video_path):
        raise ValueError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Set video capture properties for better performance
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            capture_monitor.update()
            
            # Process frame
            processed_frame, detections = process_frame(frame)
            
            # Store data in Redis if available (non-blocking)
            if redis_client and detections:
                try:
                    await redis_client.xadd(
                        f"{STREAM_PREFIX}:frames",
                        {"data": json.dumps(detections)}
                    )
                    logging_monitor.update()
                except Exception as e:
                    logger.error(f"Error storing in Redis: {e}")
            
            # Convert BGR to RGB for display
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Yield the processed frame
            yield processed_frame
    
    finally:
        cap.release()

# Redis Functions
async def init_redis():
    global redis_client, loop
    try:
        if loop is None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        redis_client = redis.from_url(
            REDIS_URL,
            socket_timeout=5,
            socket_connect_timeout=5,
            retry_on_timeout=True
        )
        await redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        redis_client = None

# FastAPI Setup
app = FastAPI(
    title="Traffic Monitoring System API",
    description="API for monitoring traffic and vehicle detection",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Root endpoint that provides API information"""
    return {
        "message": "Welcome to Traffic Monitoring System API",
        "endpoints": {
            "/fps": "Get current FPS metrics",
            "/health": "Check system health",
            "/docs": "API documentation (Swagger UI)",
            "/redoc": "API documentation (ReDoc)"
        },
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "redis_connected": redis_client is not None,
        "database_initialized": os.path.exists(DB_PATH),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/fps")
async def get_fps():
    """Get current FPS metrics for all processing stages"""
    return {
        "capture_fps": capture_monitor.get_fps(),
        "processing_fps": processing_monitor.get_fps(),
        "logging_fps": logging_monitor.get_fps(),
        "timestamp": datetime.now().isoformat()
    }

# Main Application
def main():
    global loop
    
    # Initialize database
    init_db()
    
    # Create Gradio interface
    interface = gr.Interface(
        fn=process_video_interface,
        inputs=gr.Video(),
        outputs=gr.Image(label="Processed Video", streaming=True),
        title="Traffic Monitoring System",
        description="Upload a video to process with YOLOv5 object detection",
        examples=[
            ["examples/traffic.mp4"]  # Add example video if available
        ]
    )
    
    # Start FastAPI server in a separate thread
    def run_fastapi():
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="info",
            reload=False
        )
        server = uvicorn.Server(config)
        server.run()
    
    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.daemon = True
    fastapi_thread.start()
    
    # Initialize Redis
    if loop is None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(init_redis())
    
    # Launch Gradio interface - this will block until the interface is closed
    interface.launch(
        share=False,  # Disable sharing
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        inbrowser=True  # Open in default browser
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        shutdown_event.set()
    except Exception as e:
        logger.error(f"Application error: {e}")
        shutdown_event.set()
    finally:
        if loop is not None:
            loop.close() 