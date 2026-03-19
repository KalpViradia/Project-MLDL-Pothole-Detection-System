import os
import io
import base64
import time
import uuid
import tempfile
import traceback
import logging

import requests
import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO

# ── Logging Setup ────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Global PyTorch Settings ──────────────────────────────────────────────────

torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)

# ── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="Pothole Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model Loading ────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best.pt")
model = None

@app.on_event("startup")
def load_model():
    global model
    logger.info(f"Loading YOLO model from {MODEL_PATH} ...")
    try:
        model = YOLO(MODEL_PATH)
        if torch.cuda.is_available():
            model.to("cuda")
            logger.info("Model loaded onto CUDA successfully!")
        else:
            model.to("cpu")
            logger.info("Model loaded onto CPU successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error(traceback.format_exc())

# ── In-memory report store (simulated DB) ────────────────────────────────────

reports = []

# ── Helpers ──────────────────────────────────────────────────────────────────

def _bytes_to_cv2(raw_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the uploaded image. Supported: JPEG, PNG, WebP, BMP, TIFF.")
    return img

def _image_to_base64(img_array: np.ndarray) -> str:
    success, buffer = cv2.imencode(".jpg", img_array, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise RuntimeError("Failed to encode image to JPEG")
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"

def _draw_detections(img: np.ndarray, results) -> np.ndarray:
    annotated = img.copy()
    h, w = img.shape[:2]
    ref = min(h, w)
    thickness = max(1, int(ref / 400))
    font_scale = max(0.3, min(0.7, ref / 1400))
    font_thickness = max(1, int(ref / 600))
    pad_y = max(4, int(ref / 120))
    pad_x = max(2, int(ref / 200))

    if model is None:
        return annotated

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"{model.names[cls_id]} {conf:.0%}"

            color = (0, 94, 255)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(annotated, (x1, y1 - th - pad_y), (x1 + tw + pad_x, y1), color, -1)
            cv2.putText(
                annotated, label, (x1 + pad_x // 2, y1 - pad_y // 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA,
            )

    return annotated

def _parse_detections(results) -> list:
    detections = []
    if model is None:
        return detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                "label": model.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 4),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            })
    return detections

# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    device_str = "GPU" if torch.cuda.is_available() else "CPU"
    if model is None:
        return JSONResponse({"status": "starting", "model": MODEL_PATH, "device": device_str}, status_code=503)
    return {"status": "ok", "model": MODEL_PATH, "device": device_str}


@app.post("/api/detect")
async def detect(
    request: Request,
    file: UploadFile = File(None),
    image_url: str = Form(None)
):
    try:
        img_bytes = None
        
        if file and file.filename:
            img_bytes = await file.read()
        elif image_url:
            pass
        else:
            try:
                body = await request.json()
                if isinstance(body, dict) and "image_url" in body:
                    image_url = body["image_url"]
            except Exception:
                pass
                
        if img_bytes is None and image_url:
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
                }
                resp = requests.get(image_url, headers=headers, timeout=10)
                resp.raise_for_status()
                img_bytes = resp.content
            except Exception as e:
                return JSONResponse({"error": f"Failed to fetch image URL: {str(e)}", "details": traceback.format_exc()}, status_code=400)
                
        if not img_bytes:
            return JSONResponse({"error": "No file uploaded or image_url provided.", "details": "N/A"}, status_code=400)

        img_bgr = _bytes_to_cv2(img_bytes)

        if model is None:
            return JSONResponse({"error": "Model not loaded yet.", "details": "N/A"}, status_code=503)

        start = time.perf_counter()
        results = model.predict(source=img_bgr, verbose=False, conf=0.35, iou=0.45)
        inference_time = round(time.perf_counter() - start, 3)

        detections = _parse_detections(results)
        annotated = _draw_detections(img_bgr, results)
        annotated_b64 = _image_to_base64(annotated)

        return {
            "detections": detections,
            "count": len(detections),
            "inference_time": inference_time,
            "annotated_image": annotated_b64,
            "image_size": {"width": img_bgr.shape[1], "height": img_bgr.shape[0]},
        }

    except Exception as e:
        logger.error(f"Error in /api/detect: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse({"error": str(e), "details": traceback.format_exc()}, status_code=500)


@app.post("/api/detect-video")
async def detect_video(file: UploadFile = File(None), sample_rate: int = Form(15)):
    if not file or not file.filename:
        return JSONResponse({"error": "No file uploaded. Send a video as 'file'.", "details": "N/A"}, status_code=400)

    if model is None:
        return JSONResponse({"error": "Model not loaded yet.", "details": "N/A"}, status_code=503)

    tmp_path = None
    try:
        import shutil
        suffix = os.path.splitext(file.filename)[1] or ".mp4"
        tmp_dir = tempfile.gettempdir()
        tmp_name = f"video_{int(time.time()*1000)}{suffix}"
        tmp_path = os.path.join(tmp_dir, tmp_name)
        
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return JSONResponse({"error": "Could not open video file.", "details": "N/A"}, status_code=400)

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frames_result = []
            frame_idx = 0
            total_potholes = 0
            total_inference = 0.0
            frames_with_images = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_rate == 0:
                    max_w = 960
                    if frame.shape[1] > max_w:
                        scale = max_w / frame.shape[1]
                        new_w = max_w
                        new_h = int(frame.shape[0] * scale)
                        frame = cv2.resize(frame, (new_w, new_h))

                    try:
                        start = time.perf_counter()
                        
                        device_str = 0 if torch.cuda.is_available() else "cpu"
                        results = model.predict(
                            source=frame,
                            imgsz=640,
                            conf=0.35,
                            iou=0.45,
                            half=torch.cuda.is_available(),
                            device=device_str,
                            verbose=False
                        )
                        elapsed = float(time.perf_counter() - start)
                        total_inference += float(round(elapsed, 3))

                        detections = _parse_detections(results)
                        total_potholes += int(len(detections))

                        annotated = _draw_detections(frame, results)
                        annotated_b64 = _image_to_base64(annotated)

                        val = float(frame_idx) / float(fps)
                        timestamp = float(f"{val:.2f}")

                        frames_result.append({
                            "frame_number": frame_idx,
                            "timestamp": timestamp,
                            "detections": detections,
                            "count": len(detections),
                            "inference_time": elapsed,
                            "annotated_image": annotated_b64,
                        })
                    except Exception as frame_err:
                        logger.warning(f"Skipping frame {frame_idx} due to error: {frame_err}")
                        logger.warning(traceback.format_exc())
                    finally:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                frame_idx += 1
        finally:
            cap.release()

        return {
            "frames": frames_result,
            "total_potholes": total_potholes,
            "total_frames_analyzed": len(frames_result),
            "total_video_frames": total_video_frames,
            "video_fps": float(f"{float(fps):.2f}"),
            "total_inference_time": float(f"{float(total_inference):.3f}"),
        }

    except Exception as e:
        logger.error(f"Error in /api/detect-video: {e}")
        logger.error(traceback.format_exc())

        if torch.cuda.is_available():
            logger.error(f"[DEBUG] CUDA memory allocated: {torch.cuda.memory_allocated()}")
            logger.error(f"[DEBUG] CUDA memory reserved: {torch.cuda.memory_reserved()}")

        return JSONResponse({"error": f"Server Error: {str(e)}", "details": traceback.format_exc()}, status_code=500)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@app.post("/api/report")
async def create_report(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "JSON body required.", "details": "N/A"}, status_code=400)

    if not data:
        return JSONResponse({"error": "JSON body required.", "details": "N/A"}, status_code=400)

    lat = data.get("latitude")
    lng = data.get("longitude")
    if lat is None or lng is None:
        return JSONResponse({"error": "latitude and longitude are required.", "details": "N/A"}, status_code=400)

    report = {
        "id": str(uuid.uuid4())[:8],
        "latitude": float(lat),
        "longitude": float(lng),
        "count": data.get("count", 0),
        "confidence_avg": data.get("confidence_avg", 0),
        "thumbnail": data.get("thumbnail", ""),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    reports.append(report)
    return JSONResponse(content={"ok": True, "report": report}, status_code=201)


@app.get("/api/reports")
def get_reports():
    return {"reports": reports, "total": len(reports)}
