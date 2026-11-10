import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
if not hasattr(np, "trapz"):              # NumPy 2.x compatibility, harmless otherwise
    np.trapz = np.trapezoid

import cv2
from ultralytics import YOLO

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}
BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}

# Classes to ignore during inference (case, spaces, underscores and hyphens don't matter)
EXCLUDE_CLASSES = ["wall", "box", "jar_kettle"]


def norm(s):
    return s.lower().replace(" ", "").replace("_", "").replace("-", "")


def next_dir(base: Path, name: str) -> Path:
    """runs/infer/exp, exp2, exp3 ... so nothing gets overwritten."""
    d = base / name
    i = 2
    while d.exists():
        d = base / f"{name}{i}"
        i += 1
    d.mkdir(parents=True)
    return d


def open_camera(index, backend, width, height):
    """Open a local camera, trying several Windows backends until one gives real frames."""
    order = ["dshow", "msmf", "any"] if backend == "auto" else [backend]
    for b in order:
        cap = cv2.VideoCapture(index, BACKENDS[b])
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, _ = cap.read()
        if ok:
            print(f"Camera {index} opened with backend: {b}")
            return cap
        cap.release()
    return None


def main():
    p = argparse.ArgumentParser(description="Transparent object detection inference")
    p.add_argument("--weights", default=r"D:\CV\Tranparent_Obj_Det\Models\best_untill_epoch35.pt")
    p.add_argument("--source", required=True,
                   help="image path | folder | video file | camera index (e.g. 1) | rtsp/http stream")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=640, help="use the size you trained with")
    p.add_argument("--device", default="0", help="0 for GPU, cpu for CPU")
    p.add_argument("--half", action="store_true", help="FP16 (GPU only, faster)")
    p.add_argument("--show", action="store_true", help="display results in a window")
    p.add_argument("--nosave", action="store_true", help="do not save outputs")
    p.add_argument("--save", action="store_true",
                   help="also save live (webcam/stream) output; live sources are NOT saved by default")
    p.add_argument("--flip", action="store_true", help="mirror the live view horizontally")
    p.add_argument("--backend", default="auto", choices=["auto", "dshow", "msmf", "any"],
                   help="camera backend (Windows); auto tries dshow, msmf, any")
    p.add_argument("--width", type=int, default=1280, help="camera capture width")
    p.add_argument("--height", type=int, default=720, help="camera capture height")
    p.add_argument("--winsize", default=None,
                   help="preview window size as WIDTHxHEIGHT, e.g. 1280x720")
    p.add_argument("--fullscreen", action="store_true", help="show the preview fullscreen")
    p.add_argument("--out", default="runs/infer", help="output base folder")
    p.add_argument("--name", default="exp", help="output run name")
    a = p.parse_args()

    device = "cpu" if a.device == "cpu" else int(a.device)
    model = YOLO(a.weights)

    # ---- class filtering: ignore EXCLUDE_CLASSES ----
    excl = {norm(x) for x in EXCLUDE_CLASSES}
    model_names = {norm(n) for n in model.names.values()}
    keep = [i for i, n in model.names.items() if norm(n) not in excl]
    missing = [x for x in EXCLUDE_CLASSES if norm(x) not in model_names]

    print("Model classes:", model.names)
    if missing:
        print(f"Warning: these excluded names were not found in the model: {missing}")
    print("Ignoring:", [n for i, n in model.names.items() if i not in keep])

    pred_kw = dict(conf=a.conf, iou=a.iou, imgsz=a.imgsz, device=device, half=a.half, verbose=False,
                   classes=keep if len(keep) < len(model.names) else None)

    # ---- work out what kind of source this is ----
    src = a.source
    live = False
    if src.isdigit():
        kind, source, live = "webcam", int(src), True
    elif src.lower().startswith(("rtsp://", "http://", "https://")):
        kind, source, live = "stream", src, True
    elif Path(src).is_dir():
        kind, source = "folder", src
    elif Path(src).suffix.lower() in VID_EXT:
        kind, source = "video", src
    elif Path(src).suffix.lower() in IMG_EXT:
        kind, source = "image", src
    else:
        raise SystemExit(f"Cannot tell what '{src}' is. Give an image, folder, video, or camera index.")

    show = a.show or live                              # live sources are always displayed
    save = (not a.nosave) and (not live or a.save)     # live: no saving unless --save is given
    out = next_dir(Path(a.out), a.name) if save else None

    # ---- images and folders ----
    if kind in ("image", "folder"):
        n_img = 0
        for r in model.predict(source=source, stream=True, **pred_kw):
            n_img += 1
            names = [model.names[int(c)] for c in r.boxes.cls]
            print(f"{Path(r.path).name}: {len(names)} detections {dict(Counter(names))}")
            annotated = r.plot()
            if save:
                cv2.imwrite(str(out / Path(r.path).name), annotated)
            if show:
                cv2.imshow("Detection (any key = next, q = quit)", annotated)
                if cv2.waitKey(0) & 0xFF in (ord("q"), 27):
                    break
        print(f"Processed {n_img} image(s)." + (f" Saved to: {out}" if save else ""))

    # ---- video file, webcam, stream: we read frames ourselves ----
    else:
        if kind == "webcam":
            cap = open_camera(source, a.backend, a.width, a.height)
        else:
            cap = cv2.VideoCapture(source)
        if cap is None or not cap.isOpened():
            raise SystemExit(f"Could not open source '{src}'. Try another index, or --backend msmf / dshow.")

        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        vid_fps = vid_fps if vid_fps and 1 < vid_fps < 240 else 30

        writer, t_prev, frames, fails = None, time.time(), 0, 0
        WIN = "Live Detection (q = quit, f = fullscreen)"
        if show:
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            if a.fullscreen:
                cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            elif a.winsize:
                ww, wh = map(int, a.winsize.lower().split("x"))
                cv2.resizeWindow(WIN, ww, wh)

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    if kind == "video":            # end of video file
                        break
                    fails += 1                     # camera/stream hiccup: tolerate a few
                    if fails > 50:
                        print("Camera stopped delivering frames.")
                        break
                    continue
                fails = 0

                if a.flip:
                    frame = cv2.flip(frame, 1)

                r = model.predict(frame, **pred_kw)[0]
                annotated = r.plot()
                frames += 1

                now = time.time()
                fps = 1.0 / max(now - t_prev, 1e-6)
                t_prev = now
                cv2.putText(annotated, f"FPS: {fps:.1f}  objects: {len(r.boxes)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                if save:
                    if writer is None:
                        h, w = annotated.shape[:2]
                        fname = "webcam.mp4" if kind != "video" else Path(src).stem + ".mp4"
                        writer = cv2.VideoWriter(str(out / fname),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), vid_fps, (w, h))
                    writer.write(annotated)

                if show:
                    cv2.imshow(WIN, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("f"):            # toggle fullscreen
                        full = cv2.getWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
                        cv2.setWindowProperty(
                            WIN, cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_NORMAL if full else cv2.WINDOW_FULLSCREEN)
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        print(f"Processed {frames} frame(s)." + (f" Saved to: {out}" if save else ""))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()