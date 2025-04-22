import asyncio
import redis.asyncio as redis
import logging
import time
import cv2
import uvicorn
import torch
import numpy as np
import sqlite3
import json
import os
from datetime import datetime
from fastapi import FastAPI
from starlette.responses import JSONResponse, StreamingResponse, HTMLResponse
from contextlib import asynccontextmanager
import signal
from fastapi.responses import FileResponse
import sys
import io
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HOV-Production-System")

# Redis Setup
REDIS_URL = "redis://127.0.0.1:6379"  # Use 127.0.0.1 instead of localhost
STREAM_PREFIX = "hov"
GROUP_NAME = "hov_group"
CONSUMER_NAME = f"consumer-{int(time.time())}"

# Database Setup
DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "traffic_data.db")

def ensure_db_dir():
    """Ensure the database directory exists"""
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
        logger.info(f"Created database directory: {DB_DIR}")

def get_db_connection():
    """Get a database connection"""
    ensure_db_dir()
    return sqlite3.connect(DB_PATH)

def init_db():
    """Initialize the database with required tables"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create vehicle_counts table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS vehicle_counts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        vehicle_count INTEGER,
        frame_id TEXT,
        video_source TEXT
    )
    ''')
    
    # Create index on timestamp for faster queries
    cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_timestamp 
    ON vehicle_counts(timestamp)
    ''')
    
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at: {DB_PATH}")

def store_vehicle_count(vehicle_count, frame_id, video_source):
    """Store vehicle count in the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO vehicle_counts (vehicle_count, frame_id, video_source) VALUES (?, ?, ?)",
            (vehicle_count, frame_id, video_source)
        )
        conn.commit()
        logger.info(f"Stored vehicle count {vehicle_count} for frame {frame_id}")
    except Exception as e:
        logger.error(f"Error storing vehicle count: {e}")
        raise
    finally:
        conn.close()

def get_vehicle_counts(limit=100, start_time=None, end_time=None):
    """Get vehicle counts from the database with optional time filtering"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        query = """
            SELECT timestamp, vehicle_count, frame_id, video_source 
            FROM vehicle_counts 
        """
        params = []
        
        if start_time or end_time:
            conditions = []
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time)
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time)
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        return [
            {
                "timestamp": row[0],
                "vehicle_count": row[1],
                "frame_id": row[2],
                "video_source": row[3]
            }
            for row in results
        ]
    finally:
        conn.close()

# Initialize database
init_db()

# Load YOLOv5 model
model = torch.hub.load('C:\\Users\\24793/.cache\\torch\\hub\\ultralytics_yolov5_master', 
                       'custom', path='resources/yolov5s.pt', source='local')
# Set model to evaluation mode
model.eval()
# Use newer autocast syntax
model.amp = True
# Define vehicle classes (car, truck, bus, motorcycle)
VEHICLE_CLASSES = [2, 5, 7, 3]  # COCO class IDs for vehicles

# FastAPI App
app = FastAPI()

# Initialize Redis client
redis_client = None

# Global variable to store the latest frame
latest_frame = None
latest_detections = []

# Update the frame processing simulation
async def simulate_frame_processing():
    global latest_frame, latest_detections
    # Open the video file
    cap = cv2.VideoCapture("C:\\Abhishek Data\\VSCode_Workspace\\Python\\Traffic_App\\uploads\\video.mp4")
    
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                # Reset video to start if we reach the end
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            # Perform vehicle detection with newer autocast
            with torch.amp.autocast('cuda', enabled=model.amp):
                results = model(frame)
            
            # Process detections
            detections = []
            for *xyxy, conf, cls in results.xyxy[0]:
                if int(cls) in VEHICLE_CLASSES and conf > 0.5:  # Only consider high-confidence vehicle detections
                    x1, y1, x2, y2 = map(int, xyxy)
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'class': int(cls),
                        'class_name': model.names[int(cls)]
                    })
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Add label
                    label = f"{model.names[int(cls)]} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Store the latest frame and detections
            latest_frame = frame
            latest_detections = detections
            
            # Update FPS counter
            fps_monitor.update()
            await asyncio.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            logger.error(f"Error in frame processing: {e}")
            await asyncio.sleep(1)
        finally:
            if cap is not None:
                cap.release()

@app.on_event("startup")
async def startup_event():
    global redis_client
    try:
        # Initialize Redis only once
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

    # Start the frame processing simulation
    try:
        asyncio.create_task(simulate_frame_processing())
    except Exception as e:
        logger.error(f"Failed to start frame processing: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed")

# Favicon endpoint
@app.get("/favicon.ico")
async def favicon():
    return FileResponse("resources/favicon.ico")

# Create templates directory
TEMPLATES_DIR = "templates"
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Mount templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# HTML template for the dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Traffic Monitoring Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }
        .video-container {
            background: white;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .video-feed {
            width: 100%;
            height: auto;
            border-radius: 4px;
        }
        .stats-container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .monitoring {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        .stat {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }
        .label {
            font-size: 14px;
            color: #7f8c8d;
        }
        .controls {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        button {
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            background: #3498db;
            color: white;
            cursor: pointer;
        }
        button:hover {
            background: #2980b9;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="video-container">
            <img src="/video_feed" class="video-feed" alt="Video Feed">
            <div class="controls">
                <button onclick="togglePlay()" id="playButton">Pause</button>
                <button onclick="resetVideo()">Reset</button>
            </div>
        </div>
        <div class="stats-container">
            <div class="card">
                <h3>System Status</h3>
                <div class="monitoring">
                    <div>
                        <div class="stat" id="fps">0</div>
                        <div class="label">FPS</div>
                    </div>
                    <div>
                        <div class="stat" id="uptime">0</div>
                        <div class="label">Uptime (s)</div>
                    </div>
                </div>
            </div>
            <div class="card">
                <h3>Vehicle Counts</h3>
                <div id="vehicleCounts">
                    Loading...
                </div>
            </div>
        </div>
    </div>

    <script>
        let isPlaying = true;
        const videoFeed = document.querySelector('.video-feed');
        const playButton = document.getElementById('playButton');
        
        function togglePlay() {
            isPlaying = !isPlaying;
            videoFeed.style.animationPlayState = isPlaying ? 'running' : 'paused';
            playButton.textContent = isPlaying ? 'Pause' : 'Play';
        }
        
        function resetVideo() {
            videoFeed.src = '/video_feed?' + new Date().getTime();
        }
        
        // Update monitoring data every second
        async function updateMonitoring() {
            try {
                const response = await fetch('/monitor');
                const data = await response.json();
                document.getElementById('fps').textContent = data.fps.toFixed(1);
                document.getElementById('uptime').textContent = data.system_info.uptime_seconds.toFixed(1);
            } catch (error) {
                console.error('Error updating monitoring:', error);
            }
        }
        
        // Update vehicle counts every 5 seconds
        async function updateVehicleCounts() {
            try {
                const response = await fetch('/vehicle_counts?limit=5');
                const data = await response.json();
                const countsHtml = data.counts.map(count => 
                    `<div>${new Date(count.timestamp).toLocaleTimeString()}: ${count.vehicle_count} vehicles</div>`
                ).join('');
                document.getElementById('vehicleCounts').innerHTML = countsHtml;
            } catch (error) {
                console.error('Error updating vehicle counts:', error);
            }
        }
        
        // Start periodic updates
        setInterval(updateMonitoring, 1000);
        setInterval(updateVehicleCounts, 5000);
        updateMonitoring();
        updateVehicleCounts();
    </script>
</body>
</html>
"""

# Add dashboard endpoint
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return HTMLResponse(content=DASHBOARD_HTML)

# Update root endpoint to redirect to dashboard
@app.get("/")
async def root():
    return {
        "message": "Welcome to Traffic Monitoring System",
        "endpoints": {
            "/dashboard": "Interactive dashboard",
            "/video_feed": "Live video stream",
            "/vehicle_counts": "Get vehicle count data",
            "/monitor": "Monitor system status"
        },
        "status": "operational",
        "mode": "basic"
    }

# FPS Monitor
class FPSMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.frames = 0
        self.last_update = time.time()
        self.current_fps = 0

    def update(self):
        self.frames += 1
        current_time = time.time()
        elapsed = current_time - self.last_update
        
        # Update FPS every second
        if elapsed >= 1.0:
            self.current_fps = self.frames / elapsed
            self.frames = 0
            self.last_update = current_time

    def fps(self):
        return round(self.current_fps, 2)

fps_monitor = FPSMonitor()

# Graceful Shutdown Flag
shutdown_event = asyncio.Event()

# Async Context Manager for Camera
@asynccontextmanager
async def open_camera(url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        logger.error("Failed to open camera stream.")
        yield None
    else:
        try:
            yield cap
        finally:
            cap.release()
            logger.info("Camera released.")

# Redis Stream Producer
async def camera_producer():
    global redis_client
    async with open_camera("C:\\Abhishek Data\\VSCode_Workspace\\Python\\Traffic_App\\uploads\\video.mp4") as cap:
        if cap is None:
            return

        while not shutdown_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame grab failed, retrying...")
                await asyncio.sleep(1)
                continue

            _, buffer = cv2.imencode('.jpg', frame)
            await redis_client.xadd(f"{STREAM_PREFIX}:frames", {"data": buffer.tobytes()})
            fps_monitor.update()
            await asyncio.sleep(0)  # Yield control

# Redis Stream Consumer Template
async def stream_consumer(stream_name, handler_func):
    global redis_client
    try:
        await redis_client.xgroup_create(stream_name, GROUP_NAME, id="$", mkstream=True)
    except redis.ResponseError:
        pass  # Group already exists

    while not shutdown_event.is_set():
        try:
            response = await redis_client.xreadgroup(
                groupname=GROUP_NAME,
                consumername=CONSUMER_NAME,
                streams={stream_name: '>'},
                count=1,
                block=1000
            )

            if response:
                for stream, messages in response:
                    for msg_id, msg in messages:
                        await handler_func(msg_id, msg)

        except Exception as e:
            logger.exception(f"Error in consumer for {stream_name}: {e}")
            await asyncio.sleep(1)

# Vehicle Detection Handler
async def vehicle_detection_handler(msg_id, msg):
    try:
        # Decode the frame from Redis message
        frame_data = msg[b"data"]
        frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error("Failed to decode frame")
            return

        # Perform inference
        results = model(frame)
        
        # Filter for vehicles and format detections
        vehicle_detections = []
        for *xyxy, conf, cls in results.xyxy[0]:
            if int(cls) in VEHICLE_CLASSES and conf > 0.5:  # Only consider high-confidence vehicle detections
                x1, y1, x2, y2 = map(int, xyxy)
                vehicle_data = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'class': int(cls),
                    'class_name': model.names[int(cls)]
                }
                vehicle_detections.append(vehicle_data)

        # Serialize detections and send to next stream
        detection_data = json.dumps(vehicle_detections).encode('utf-8')
        await redis_client.xadd(f"{STREAM_PREFIX}:vehicles", {"data": detection_data})
        
        # Acknowledge the message
        await redis_client.xack(f"{STREAM_PREFIX}:frames", GROUP_NAME, msg_id)
        
        logger.info(f"Processed frame with {len(vehicle_detections)} vehicle detections")
        
    except Exception as e:
        logger.exception(f"Error in vehicle detection: {e}")
        # Don't acknowledge the message if there was an error
        # This allows for retry processing

# Handler Functions for Each Module
async def tracking_handler(msg_id, msg):
    # TODO: Replace with real tracking
    tracked_data = b"dummy_tracking_data"
    await redis_client.xadd(f"{STREAM_PREFIX}:tracked", {"data": tracked_data})
    await redis_client.xack(f"{STREAM_PREFIX}:vehicles", GROUP_NAME, msg_id)

async def face_detection_handler(msg_id, msg):
    # TODO: Replace with real face detection
    faces_data = b"dummy_faces_data"
    await redis_client.xadd(f"{STREAM_PREFIX}:faces", {"data": faces_data})
    await redis_client.xack(f"{STREAM_PREFIX}:tracked", GROUP_NAME, msg_id)

async def plate_detection_handler(msg_id, msg):
    # TODO: Replace with real plate detection
    plate_data = b"dummy_plate_data"
    await redis_client.xadd(f"{STREAM_PREFIX}:plates", {"data": plate_data})
    await redis_client.xack(f"{STREAM_PREFIX}:faces", GROUP_NAME, msg_id)

async def db_storage_handler(msg_id, msg):
    try:
        # Decode the plate data from Redis message
        plate_data = msg[b"data"]
        detections = json.loads(plate_data.decode('utf-8'))
        
        # Get vehicle count
        vehicle_count = len(detections)
        
        # Store in database
        store_vehicle_count(
            vehicle_count=vehicle_count,
            frame_id=msg_id.decode('utf-8'),
            video_source="video.mp4"  # You can make this dynamic based on your needs
        )
        
        # Acknowledge the message
        await redis_client.xack(f"{STREAM_PREFIX}:plates", GROUP_NAME, msg_id)
        
    except Exception as e:
        logger.exception(f"Error storing data in database: {e}")
        # Don't acknowledge the message if there was an error
        # This allows for retry processing

# Update the vehicle_counts endpoint to use detections
@app.get("/vehicle_counts")
async def get_vehicle_counts(
    limit: int = 100,
    start_time: str = None,
    end_time: str = None
):
    try:
        # Update FPS counter
        fps_monitor.update()
        
        # Get counts from database
        counts = get_vehicle_counts(limit, start_time, end_time)
        
        # Add current detection count
        current_count = len(latest_detections)
        
        return JSONResponse({
            "counts": counts,
            "total_records": len(counts),
            "status": "success",
            "processing_info": {
                "fps": fps_monitor.fps(),
                "last_update": datetime.now().isoformat(),
                "current_detections": current_count,
                "detection_details": latest_detections
            }
        })
    except Exception as e:
        logger.error(f"Error getting vehicle counts: {e}")
        return JSONResponse(
            {"error": "Failed to get vehicle counts"},
            status_code=500
        )

# FastAPI Monitoring Endpoint
@app.get("/monitor")
async def monitor():
    try:
        # Update FPS counter
        fps_monitor.update()
        
        # Basic system info without Redis
        return JSONResponse({
            "fps": fps_monitor.fps(),
            "status": "operational",
            "message": "Running in basic mode (Redis not available)",
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": round(time.time() - fps_monitor.start_time, 2)
            },
            "performance": {
                "total_frames_processed": fps_monitor.frames,
                "average_fps": fps_monitor.fps()
            }
        })
    except Exception as e:
        logger.error(f"Error in monitor endpoint: {e}")
        return JSONResponse(
            {"error": "Failed to get monitoring data"},
            status_code=500
        )

# Shutdown Handler
def shutdown():
    shutdown_event.set()

# Add video streaming endpoint
@app.get("/video_feed")
async def video_feed():
    async def generate():
        while True:
            if latest_frame is not None:
                # Convert frame to JPEG
                ret, buffer = cv2.imencode('.jpg', latest_frame)
                frame_bytes = buffer.tobytes()
                
                # Yield the frame in the response
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            await asyncio.sleep(0.033)  # Match the frame rate

    return StreamingResponse(
        generate(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

# Main Runner
async def main():
    # Remove Redis initialization since it's now handled in startup_event
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, shutdown)
    loop.add_signal_handler(signal.SIGTERM, shutdown)

    try:
        tasks = [
            camera_producer(),
            stream_consumer(f"{STREAM_PREFIX}:frames", vehicle_detection_handler),
            stream_consumer(f"{STREAM_PREFIX}:vehicles", tracking_handler),
            stream_consumer(f"{STREAM_PREFIX}:tracked", face_detection_handler),
            stream_consumer(f"{STREAM_PREFIX}:faces", plate_detection_handler),
            stream_consumer(f"{STREAM_PREFIX}:plates", db_storage_handler),
        ]
        await asyncio.gather(*tasks)
    except Exception as e:
        logger.error(f"Error in main task: {e}")
        raise

if __name__ == "__main__":
    try:
        # Create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Create the main task
        main_task = loop.create_task(main())
        
        # Run the server
        config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="info")
        server = uvicorn.Server(config)
        loop.run_until_complete(server.serve())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        # Cleanup
        if 'main_task' in locals():
            main_task.cancel()
        loop.close()
