from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from ultralytics import YOLO
import cv2
import numpy as np
import asyncio
import time
import os
from pathlib import Path
import websockets.exceptions

# === 📁 Path Setup ===
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
MODEL_PATH = BASE_DIR / "best.pt"

# === 🚀 Init FastAPI ===
app = FastAPI(
    title="Douxelle QC System",
    description="Premium Quality Control Detection",
    version="3.0"
)

# === 🌐 CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

import requests
from tqdm import tqdm

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    total = int(response.headers.get('content-length', 0))
    with open(destination, "wb") as f, tqdm(
        desc="⬇️ Downloading model",
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

MODEL_ID = "1MzNcsebrm0IBUhbsRoD7L8tW3Grv85MJ"
MODEL_PATH = BASE_DIR / "best.pt"

if not MODEL_PATH.exists():
    download_file_from_google_drive(MODEL_ID, MODEL_PATH)
    print("✅ Model downloaded successfully")



model = YOLO(str(MODEL_PATH))
print(f"🚀 Model loaded successfully from: {MODEL_PATH}")

# === 🌐 Check frontend folder ===
if not FRONTEND_DIR.exists():
    raise RuntimeError(f"Frontend folder not found at: {FRONTEND_DIR}")
else:
    print(f"🌐 Frontend served from: {FRONTEND_DIR}")

# === 📡 WebSocket Manager ===
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

manager = ConnectionManager()

# === 🧪 WebSocket Endpoint ===
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            try:
                start_time = time.time()
                data = await websocket.receive_bytes()
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    await websocket.send_json({"error": "Gambar tidak dapat diproses"})
                    continue

                results = model.predict(img, imgsz=320, conf=0.5, stream=True)
                result = next(results)

                await websocket.send_json({
                    "boxes": result.boxes.xyxy.tolist(),
                    "classes": result.boxes.cls.tolist(),
                    "confidences": result.boxes.conf.tolist(),
                    "inference_time": (time.time() - start_time) * 1000
                })

            except asyncio.IncompleteReadError:
                break
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"⚠️ WebSocket closed: {e}")
                break
            except Exception as e:
                print(f"🚨 WebSocket error: {e}")
                break
    finally:
        manager.disconnect(websocket)
        print("🔌 WebSocket client disconnected")

# === 📤 Upload Detection Endpoint ===
@app.post("/api/detect")
async def detect_from_upload(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="File kosong")

        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Format file tidak dikenali")

        # Get original image dimensions
        height, width = img.shape[:2]
        
        results = model.predict(img, imgsz=640, conf=0.5)
        result = results[0]

        detections = []
        counts = {"layak": 0, "tidak_layak": 0}

        for box in result.boxes:
            cls = int(box.cls[0])
            xyxy = box.xyxy[0].tolist()
            
            # Ensure coordinates are within image bounds
            xyxy = [
                max(0, min(width, xyxy[0])),
                max(0, min(height, xyxy[1])),
                max(0, min(width, xyxy[2])),
                max(0, min(height, xyxy[3]))
            ]
            
            detections.append({
                "class": cls,
                "confidence": float(box.conf[0]),
                "coordinates": xyxy
            })
            counts["layak" if cls == 0 else "tidak_layak"] += 1

        return JSONResponse({
            "detections": detections,
            "counts": counts,
            "image_size": {"width": width, "height": height}
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# === 🌐 Serve HTML pages ===
@app.get("/detect-realtime")
async def get_detect_realtime():
    filepath = FRONTEND_DIR / "detect-realtime.html"
    if filepath.exists():
        return FileResponse(filepath)
    raise HTTPException(status_code=404, detail="File detect-realtime.html tidak ditemukan")

# === 🗂️ Serve Static Frontend Files ===
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

# === ▶️ Run Dev Server ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
