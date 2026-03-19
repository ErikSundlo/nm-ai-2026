"""
Task 1 — NorgesGruppen Grocery Detection Agent
POST /solve  →  run YOLO inference on image, return detections

Expected request body (JSON):
{
    "image": "<base64-encoded image>",        // primary format
    "images": ["<base64>", ...]               // batch format (optional)
}

OR multipart/form-data with field "image"

Response:
{
    "status": "completed",
    "detections": [
        {
            "bbox": [x, y, w, h],             // COCO format (abs pixels)
            "category_id": <int>,
            "category_name": "<str>",
            "score": <float>
        },
        ...
    ]
}
"""
import base64
import io
import json
import logging
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Load model — prefer best.pt if exists, fall back to last.pt
MODEL_PATHS = [
    Path("runs/detect/runs/task1/groceries_nano/weights/best.pt"),
    Path("runs/detect/runs/task1/groceries_nano/weights/last.pt"),
    Path("runs/detect/runs/task1/groceries/weights/best.pt"),
    Path("runs/detect/runs/task1/groceries/weights/last.pt"),
]

_model: YOLO | None = None
_class_names: list[str] = []


def get_model() -> YOLO:
    global _model, _class_names
    if _model is None:
        for p in MODEL_PATHS:
            if p.exists():
                log.info("Loading model from %s", p)
                _model = YOLO(str(p))
                _class_names = list(_model.names.values())
                return _model
        raise RuntimeError("No trained model found. Run training first.")
    return _model


app = FastAPI(title="BradskiBeat — NorgesGruppen Detector")


@app.on_event("startup")
def startup():
    try:
        get_model()
        log.info("Model loaded on startup. Classes: %d", len(_class_names))
    except RuntimeError as e:
        log.warning("Startup model load failed: %s", e)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    log.info("INCOMING %s %s", request.method, request.url.path)
    response = await call_next(request)
    log.info("RESPONSE %s %s -> %s", request.method, request.url.path, response.status_code)
    return response


def decode_image(b64_str: str) -> Image.Image:
    """Decode a base64 image string to a PIL Image."""
    # Strip data URI prefix if present
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def run_inference(images: list[Image.Image], conf_threshold: float = 0.25) -> list[list[dict]]:
    """Run YOLO on a list of PIL images. Returns list of detection lists."""
    model = get_model()
    results = model(images, conf=conf_threshold, verbose=False)

    all_detections = []
    for result in results:
        detections = []
        if result.boxes is not None:
            boxes  = result.boxes.xyxy.cpu().numpy()   # x1,y1,x2,y2
            scores = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, cls_ids):
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                w = x2 - x1
                h = y2 - y1
                detections.append({
                    "bbox":          [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    "category_id":   int(cls_id),
                    "category_name": _class_names[cls_id] if cls_id < len(_class_names) else str(cls_id),
                    "score":         round(float(score), 4),
                })
        all_detections.append(detections)
    return all_detections


@app.get("/solve")
@app.get("/solve ")
def solve_ping():
    return {"status": "ok"}


@app.post("/solve")
@app.post("/solve ")
async def solve(request: Request):
    content_type = request.headers.get("content-type", "")

    if "multipart" in content_type:
        form = await request.form()
        img_file = form.get("image")
        if img_file is None:
            raise HTTPException(400, "No 'image' field in multipart form")
        img_bytes = await img_file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        images = [pil_img]
    else:
        body = await request.json()
        log.info("Payload keys: %s", list(body.keys()))

        if "images" in body:
            images = [decode_image(b64) for b64 in body["images"]]
        elif "image" in body:
            images = [decode_image(body["image"])]
        else:
            # Maybe there's base64 data under another key — try to find it
            log.warning("No 'image'/'images' key found. Keys: %s", list(body.keys()))
            raise HTTPException(400, f"Expected 'image' or 'images' key. Got: {list(body.keys())}")

    try:
        detections_per_image = run_inference(images)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    # Flatten for single-image requests
    if len(images) == 1:
        return {
            "status":     "completed",
            "detections": detections_per_image[0],
        }
    return {
        "status": "completed",
        "results": [{"detections": d} for d in detections_per_image],
    }


@app.get("/health")
@app.get("/")
def health():
    model_ready = any(p.exists() for p in MODEL_PATHS)
    return {"status": "ok", "team": "BradskiBeat", "model_ready": model_ready}


@app.post("/")
async def solve_root(request: Request):
    log.info("POST / — redirecting to /solve handler")
    return await solve(request)
