"""
Evaluate a trained YOLO11 model on the validation (or test) set.

Reports: precision, recall, mAP50, mAP75, mAP50-95 (overall and per class),
plus inference speed. Results are also saved to CSV and JSON.

Usage:
    python evaluate.py
    python evaluate.py --weights path/to/best.pt --data transparent_obj_det.yaml
    python evaluate.py --split test --imgsz 640 --batch 16 --device 0
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
if not hasattr(np, "trapz"):              # NumPy 2.x compatibility, harmless otherwise
    np.trapz = np.trapezoid

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description="Evaluate a YOLO model: P, R, mAP50, mAP75, mAP50-95")
    p.add_argument("--weights", default=r"D:\CV\Tranparent_Obj_Det\Models\last_4.59.pt")
    p.add_argument("--data", default=r"D:\CV\Tranparent_Obj_Det\src\yolov11\transparent_obj_det.yaml")
    p.add_argument("--split", default="val", choices=["val", "test", "train"],
                   help="which split of the dataset YAML to evaluate on")
    p.add_argument("--imgsz", type=int, default=640, help="use the size you trained with")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0", help="0 for GPU, cpu for CPU")
    p.add_argument("--conf", type=float, default=0.001,
                   help="keep this low for mAP; a high value lowers recall and mAP")
    p.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    p.add_argument("--half", action="store_true", help="FP16 evaluation (GPU only)")
    p.add_argument("--plots", action="store_true", help="save PR curve, confusion matrix, etc.")
    p.add_argument("--out", default="runs/eval", help="output base folder")
    p.add_argument("--name", default="exp", help="output run name")
    a = p.parse_args()

    device = "cpu" if a.device == "cpu" else int(a.device)
    model = YOLO(a.weights)

    res = model.val(
        data=a.data, split=a.split, imgsz=a.imgsz, batch=a.batch, device=device,
        conf=a.conf, iou=a.iou, half=a.half, plots=a.plots,
        project=a.out, name=a.name, workers=2, verbose=False,
    )
    box = res.box
    save_dir = Path(res.save_dir)

    # ---- overall metrics ----
    overall = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP75": float(box.map75),
        "mAP50-95": float(box.map),
    }

    # ---- per-class metrics (all_ap columns are IoU 0.50, 0.55, ..., 0.95) ----
    per_class = []
    for k, ci in enumerate(box.ap_class_index):
        ap = box.all_ap[k]
        per_class.append({
            "class": model.names[int(ci)],
            "precision": float(box.p[k]),
            "recall": float(box.r[k]),
            "mAP50": float(ap[0]),
            "mAP75": float(ap[5]),
            "mAP50-95": float(ap.mean()),
        })

    speed = {k: round(float(v), 2) for k, v in res.speed.items()}   # ms per image
    total_ms = sum(speed.values())

    # ---- print ----
    print("\n" + "=" * 70)
    print(f"Weights : {a.weights}")
    print(f"Dataset : {a.data}  (split = {a.split})")
    print("=" * 70)
    print(f"{'Precision':<12}{overall['precision']:.4f}")
    print(f"{'Recall':<12}{overall['recall']:.4f}")
    print(f"{'mAP50':<12}{overall['mAP50']:.4f}")
    print(f"{'mAP75':<12}{overall['mAP75']:.4f}")
    print(f"{'mAP50-95':<12}{overall['mAP50-95']:.4f}")

    print("\nPer-class results")
    header = f"{'Class':<20}{'P':>8}{'R':>8}{'mAP50':>8}{'mAP75':>8}{'mAP50-95':>10}"
    print(header)
    print("-" * len(header))
    for r in per_class:
        print(f"{r['class']:<20}{r['precision']:>8.3f}{r['recall']:>8.3f}"
              f"{r['mAP50']:>8.3f}{r['mAP75']:>8.3f}{r['mAP50-95']:>10.3f}")

    print(f"\nSpeed per image: {speed} ms "
          f"(total {total_ms:.1f} ms, about {1000 / total_ms:.1f} FPS)" if total_ms > 0 else "")

    # ---- save ----
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "metrics_overall.json", "w") as f:
        json.dump({"weights": a.weights, "data": a.data, "split": a.split,
                   "overall": overall, "speed_ms": speed}, f, indent=2)
    with open(save_dir / "metrics_per_class.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_class[0].keys()) if per_class else ["class"])
        w.writeheader()
        w.writerows(per_class)
    print(f"\nSaved to: {save_dir}")


if __name__ == "__main__":
    main()