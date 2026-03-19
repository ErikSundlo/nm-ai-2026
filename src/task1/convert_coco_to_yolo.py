"""
Convert COCO annotations to YOLO format and create train/val split.

Output layout:
  data/task1_yolo/
    dataset.yaml
    images/train/
    images/val/
    labels/train/
    labels/val/
"""
import json
import random
import shutil
from pathlib import Path

COCO_DIR   = Path("data/raw/task1/coco/train")
OUT_DIR    = Path("data/task1_yolo")
VAL_RATIO  = 0.15
SEED       = 42

random.seed(SEED)


def main():
    ann_path = COCO_DIR / "annotations.json"
    img_dir  = COCO_DIR / "images"

    with open(ann_path) as f:
        coco = json.load(f)

    images    = {img["id"]: img for img in coco["images"]}
    cat_ids   = sorted(c["id"] for c in coco["categories"])
    cat_map   = {old_id: new_id for new_id, old_id in enumerate(cat_ids)}
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}

    ann_by_img: dict[int, list] = {img_id: [] for img_id in images}
    for ann in coco["annotations"]:
        ann_by_img[ann["image_id"]].append(ann)

    all_ids = list(images.keys())
    random.shuffle(all_ids)
    n_val = max(1, int(len(all_ids) * VAL_RATIO))
    val_ids   = set(all_ids[:n_val])
    train_ids = set(all_ids[n_val:])
    print(f"Train: {len(train_ids)}  Val: {len(val_ids)}  Classes: {len(cat_ids)}")

    for split, id_set in [("train", train_ids), ("val", val_ids)]:
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

        for img_id in id_set:
            img_info = images[img_id]
            fname    = img_info["file_name"]
            W, H     = img_info["width"], img_info["height"]

            src = img_dir / fname
            dst = OUT_DIR / "images" / split / fname
            if not dst.exists():
                shutil.copy2(src, dst)

            label_path = OUT_DIR / "labels" / split / (Path(fname).stem + ".txt")
            lines = []
            for ann in ann_by_img[img_id]:
                x, y, w, h = ann["bbox"]
                if w <= 0 or h <= 0:
                    continue
                cx = max(0.0, min(1.0, (x + w / 2) / W))
                cy = max(0.0, min(1.0, (y + h / 2) / H))
                nw = max(0.0, min(1.0, w / W))
                nh = max(0.0, min(1.0, h / H))
                cls = cat_map[ann["category_id"]]
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            label_path.write_text("\n".join(lines))

    abs_out = OUT_DIR.resolve()
    names = [cat_names[old_id] for old_id in cat_ids]

    yaml_lines = [
        f"path: {abs_out.as_posix()}",
        "train: images/train",
        "val:   images/val",
        "",
        f"nc: {len(cat_ids)}",
        "names:",
    ]
    for i, name in enumerate(names):
        safe = name.replace("'", "").replace('"', '').replace('\\', '')
        yaml_lines.append(f"  {i}: '{safe}'")

    (OUT_DIR / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n")
    print(f"Dataset written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
