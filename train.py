import argparse
import numpy as np
if not hasattr(np, "trapz"):          # NumPy 2.x fix, harmless if already patched
    np.trapz = np.trapezoid
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=r"D:\CV\Tranparent_Obj_Det\src\yolov11\transparent_obj_det.yaml")
    p.add_argument("--model", default="yolo11m.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", default=None, help="path to last.pt to resume from")
    a = p.parse_args()

    # Resume branch: the saved checkpoint already holds all training settings
    if a.resume:
        YOLO(a.resume).train(resume=True)
        return

    device = "cpu" if a.device == "cpu" else int(a.device)
    model = YOLO(a.model)
    model.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        device=device, workers=a.workers, patience=30,
        project="runs/transparent", name="exp1",
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, degrees=10, fliplr=0.5,
        mosaic=1.0, close_mosaic=10,
    )

if __name__ == "__main__":
    main()