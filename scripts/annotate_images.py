#!/usr/bin/env python3
"""
Draw YOLO-format labels (YOLOv5/v8/v11 .txt) on images and save the annotated
images to another folder.

Label line formats supported:
  Detection    : class x_center y_center width height          (normalized 0-1)
  Detection+conf: class x_center y_center width height conf
  Segmentation : class x1 y1 x2 y2 x3 y3 ...                   (normalized polygon)

Usage:
  python annotate_yolo.py --images path/to/images --labels path/to/labels --output path/to/output
  python annotate_yolo.py -i images -l labels -o annotated --classes classes.txt
  python annotate_yolo.py -i images -l labels -o annotated --classes data.yaml
  python annotate_yolo.py -i images -l labels -o annotated --classes "person,car,dog"

Requires: pip install opencv-python numpy   (pyyaml only if you pass a data.yaml)
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_class_names(src):
    """Return dict {id: name}. src can be a .txt (one name per line), a data.yaml,
    or a comma-separated string. Returns {} if src is None."""
    if not src:
        return {}

    p = Path(src)
    if p.is_file():
        if p.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError:
                sys.exit("pyyaml is needed for data.yaml  ->  pip install pyyaml")
            with open(p, "r", encoding="utf-8") as f:
                names = yaml.safe_load(f).get("names", {})
            if isinstance(names, list):
                return {i: str(n) for i, n in enumerate(names)}
            return {int(k): str(v) for k, v in names.items()}
        # plain text file, one class per line
        with open(p, "r", encoding="utf-8") as f:
            return {i: line.strip() for i, line in enumerate(f) if line.strip()}

    # comma-separated string
    return {i: n.strip() for i, n in enumerate(src.split(",")) if n.strip()}


def class_color(cls_id):
    """Deterministic, distinct BGR color per class id."""
    hsv = np.uint8([[[(cls_id * 37) % 180, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def parse_label_file(path):
    """Yield (cls_id, coords_list, conf_or_None) for each valid line."""
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cls_id = int(float(parts[0]))
                vals = [float(v) for v in parts[1:]]
            except ValueError:
                print(f"  [warn] {path.name}:{ln} could not parse -> skipped")
                continue
            if len(vals) == 4:
                yield cls_id, vals, None
            elif len(vals) == 5:
                yield cls_id, vals[:4], vals[4]
            elif len(vals) >= 6 and len(vals) % 2 == 0:
                yield cls_id, vals, None  # segmentation polygon
            else:
                print(f"  [warn] {path.name}:{ln} unexpected format -> skipped")


def draw_label_text(img, text, x, y, color, thickness):
    scale = max(0.4, min(img.shape[:2]) / 1000)
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, thickness - 1))
    y_top = y - th - base
    if y_top < 0:  # keep text inside the image
        y_top = y
    cv2.rectangle(img, (x, y_top), (x + tw + 4, y_top + th + base), color, -1)
    brightness = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    txt_color = (0, 0, 0) if brightness > 160 else (255, 255, 255)
    cv2.putText(img, text, (x + 2, y_top + th), cv2.FONT_HERSHEY_SIMPLEX,
                scale, txt_color, max(1, thickness - 1), cv2.LINE_AA)


def annotate_image(img, label_path, names, thickness=None, hide_labels=False):
    h, w = img.shape[:2]
    thickness = thickness or max(2, round(min(h, w) / 400))
    count = 0

    for cls_id, vals, conf in parse_label_file(label_path):
        color = class_color(cls_id)
        text = names.get(cls_id, str(cls_id))
        if conf is not None:
            text += f" {conf:.2f}"

        if len(vals) == 4:  # bounding box
            xc, yc, bw, bh = vals
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            anchor = (max(x1, 0), max(y1, 0))
        else:  # polygon
            pts = np.array(vals, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= w
            pts[:, 1] *= h
            pts = pts.astype(np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            cv2.polylines(img, [pts], True, color, thickness)
            anchor = (int(pts[:, 0].min()), int(pts[:, 1].min()))

        if not hide_labels:
            draw_label_text(img, text, anchor[0], anchor[1], color, thickness)
        count += 1

    return img, count


def main():
    ap = argparse.ArgumentParser(description="Annotate images with YOLO labels.")
    ap.add_argument("-i", "--images", required=True, help="Folder with images")
    ap.add_argument("-l", "--labels", required=True, help="Folder with YOLO .txt labels")
    ap.add_argument("-o", "--output", required=True, help="Folder to save annotated images")
    ap.add_argument("-c", "--classes", default=None,
                    help="classes.txt, data.yaml, or comma-separated names (optional)")
    ap.add_argument("-t", "--thickness", type=int, default=None, help="Box line thickness (auto if omitted)")
    ap.add_argument("--hide-labels", action="store_true", help="Draw boxes without class text")
    args = ap.parse_args()

    img_dir, lbl_dir, out_dir = Path(args.images), Path(args.labels), Path(args.output)
    if not img_dir.is_dir():
        sys.exit(f"Images folder not found: {img_dir}")
    if not lbl_dir.is_dir():
        sys.exit(f"Labels folder not found: {lbl_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    names = load_class_names(args.classes)
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        sys.exit(f"No images found in {img_dir}")

    done = no_label = failed = total_objs = 0
    for n, img_path in enumerate(images, 1):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[{n}/{len(images)}] could not read {img_path.name} -> skipped")
            failed += 1
            continue

        if lbl_path.exists():
            img, cnt = annotate_image(img, lbl_path, names, args.thickness, args.hide_labels)
            total_objs += cnt
            done += 1
        else:
            no_label += 1  # image is still copied (unannotated) so the output stays complete
            print(f"[{n}/{len(images)}] no label file for {img_path.name} -> saved as-is")

        cv2.imwrite(str(out_dir / img_path.name), img)
        if n % 50 == 0:
            print(f"  processed {n}/{len(images)}")

    print("\n--- Summary ---")
    print(f"Images found        : {len(images)}")
    print(f"Annotated           : {done}  ({total_objs} objects drawn)")
    print(f"Without label file  : {no_label}")
    print(f"Unreadable          : {failed}")
    print(f"Saved to            : {out_dir.resolve()}")


if __name__ == "__main__":
    main()