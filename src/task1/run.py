"""
NorgesGruppen grocery detection — BradskiBeat submission.
Entry point for competition evaluation.

Usage:
    python run.py --images /path/to/images/ --output predictions.json
"""
import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "best.pt"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45
IMG_SIZE       = 416


def load_model():
    return YOLO(str(MODEL_PATH))


def predict_image(model, image_path: str, image_id=None) -> list:
    results = model(image_path, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                    imgsz=IMG_SIZE, verbose=False)
    detections = []
    for result in results:
        if result.boxes is None:
            continue
        boxes   = result.boxes.xyxy.cpu().numpy()
        scores  = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, cls_ids):
            det = {
                "bbox":        [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score":       float(score),
                "category_id": int(cls_id),
            }
            if image_id is not None:
                det["image_id"] = image_id
            detections.append(det)
    return detections


def predict_directory(model, images_dir: str) -> list:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted(
        p for p in Path(images_dir).rglob("*") if p.suffix.lower() in exts
    )
    all_detections = []
    for img_path in image_files:
        dets = predict_image(model, str(img_path), image_id=img_path.stem)
        all_detections.extend(dets)
    return all_detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, default="test/images")
    parser.add_argument("--output", type=str, default="predictions.json")
    args = parser.parse_args()

    model      = load_model()
    detections = predict_directory(model, args.images)

    Path(args.output).write_text(json.dumps(detections))
    print(f"Wrote {len(detections)} detections to {args.output}")


if __name__ == "__main__":
    main()
