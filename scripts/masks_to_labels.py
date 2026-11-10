"""
masks_rgb_to_yolo.py
--------------------
Turns COLOUR segmentation masks (Trans10K-v2 style, even if saved as JPG) into YOLO
object-detection labels (works for YOLOv8 / YOLO11).

Input : a folder of images + a folder of colour masks (same file names;
        'name_mask.jpg' / 'name_mask.png' also work)
Output: one .txt per image in the output folder, one line per object:
            class_id x_center y_center width height        (all values 0-1)

How it works
  1. every mask pixel is matched to the NEAREST colour of the palette below
     (JPG compression changes colours a little, so exact matching would fail);
     pixels far from every palette colour are treated as background;
  2. a small majority filter removes JPG speckles (--smooth);
  3. for each class, every separate blob becomes one bounding box.

YOLO class ids (background is NOT a class):
  0 box   1 bottle   2 window   3 eyeglass   4 freezer   5 jar_kettle
  6 door  7 cup      8 wall     9 bowl       10 shelf

USAGE (type on one line)
  1) test on 20 masks, nothing is written - read the report:
     python masks_rgb_to_yolo.py --images Trans10K/train/images --masks Trans10K/train/masks --inspect

  2) convert, and save 30 pictures with boxes drawn so you can check them:
     python masks_rgb_to_yolo.py --images Trans10K/train/images --masks Trans10K/train/masks --out Trans10K/train/labels --check 30

  3) use several CPU cores for the full set:
     python masks_rgb_to_yolo.py --images ... --masks ... --out ... --workers 4

YOLO finds labels by replacing the folder name 'images' with 'labels' in each image path,
so the best output folder is a 'labels' folder NEXT TO your 'images' folder.

Known weak points (look at the box_check pictures):
  * some palette colours are very close (wall/bowl, window/jar_kettle, door/eyeglass);
    heavy JPG damage can mix them up. The report tells you how noisy your masks are;
  * touching objects of the same class merge into one box;
  * one object cut in pieces by something in front of it gives several boxes (--merge-gap).
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# index 0 = background; the rest are YOLO class (index - 1)
PALETTE = [
    ("background", (0, 0, 0)),
    ("box", (4, 250, 7)),
    ("bottle", (150, 5, 61)),
    ("window", (204, 255, 4)),
    ("eyeglass", (140, 140, 140)),
    ("freezer", (6, 230, 230)),
    ("jar_kettle", (235, 255, 7)),
    ("door", (120, 120, 120)),
    ("cup", (255, 51, 7)),
    ("wall", (224, 5, 255)),
    ("bowl", (204, 5, 255)),
    ("shelf", (120, 120, 70)),
]
CLASS_NAMES = [n for n, _ in PALETTE[1:]]
PAL = np.array([c for _, c in PALETTE], np.float32)
PAL_SQ = (PAL ** 2).sum(1)


# ---------------------------------------------------------------- core logic
def classify(mask_rgb, max_dist, smooth):
    """Return (label_map uint8 [0=background, 1..11], fraction_far_from_palette)."""
    h, w, _ = mask_rgb.shape
    px = mask_rgb.reshape(-1, 3).astype(np.float32)
    d2 = (px ** 2).sum(1)[:, None] - 2.0 * (px @ PAL.T) + PAL_SQ[None, :]
    idx = d2.argmin(1)
    dmin = np.sqrt(np.maximum(np.take_along_axis(d2, idx[:, None], 1)[:, 0], 0))
    far = dmin > max_dist
    idx[far] = 0                                           # unknown colour -> background
    labels = idx.reshape(h, w).astype(np.uint8)
    if smooth and smooth > 1:                              # majority filter
        counts = np.stack([cv2.blur((labels == c).astype(np.float32), (smooth, smooth))
                           for c in range(len(PALETTE))])
        labels = counts.argmax(0).astype(np.uint8)
    return labels, float(far.mean())


def boxes_from_labels(labels, min_area, merge_gap):
    """List of (class_id, x, y, w, h) in pixels of the label map."""
    boxes = []
    for c in range(1, len(PALETTE)):
        binary = (labels == c).astype(np.uint8)
        if not binary.any():
            continue
        work = binary
        if merge_gap > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * merge_gap + 1, 2 * merge_gap + 1))
            work = cv2.dilate(binary, k)
        n, _, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
        for i in range(1, n):
            x, y, w, h, _ = stats[i]
            if int(binary[y:y + h, x:x + w].sum()) >= min_area:
                boxes.append((c - 1, int(x), int(y), int(w), int(h)))
    return boxes


def load_mask(path, max_side):
    m = np.array(Image.open(path).convert("RGB"))
    if max_side and max(m.shape[:2]) > max_side:
        s = max_side / max(m.shape[:2])
        m = cv2.resize(m, (int(m.shape[1] * s), int(m.shape[0] * s)), interpolation=cv2.INTER_NEAREST)
    return m


def process_one(job):
    img_path, mask_path, out_dir, cfg = job
    mask = load_mask(mask_path, cfg["max_side"])
    h, w = mask.shape[:2]
    labels, far = classify(mask, cfg["max_dist"], cfg["smooth"])
    min_area = max(4, int(cfg["min_area_frac"] * h * w))
    boxes = boxes_from_labels(labels, min_area, cfg["merge_gap"])

    iw, ih = Image.open(img_path).size
    aspect_bad = abs(iw / ih - w / h) > 0.02 * (iw / ih)

    lines = [f"{c} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"
             for c, x, y, bw, bh in boxes]
    if out_dir is not None:
        (Path(out_dir) / f"{Path(img_path).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    px = Counter({int(c): int((labels == c).sum()) for c in np.unique(labels)})
    return {"img": str(img_path), "boxes": [(c, x / w, y / h, bw / w, bh / h) for c, x, y, bw, bh in boxes],
            "far": far, "aspect_bad": aspect_bad, "pixels": px}


# ---------------------------------------------------------------- helpers
def build_mask_index(mask_dir):
    index = {}
    for p in Path(mask_dir).rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            stem = p.stem[:-5] if p.stem.lower().endswith("_mask") else p.stem
            index[stem] = p
    return index


def warn_close_colours():
    close = []
    for i in range(1, len(PALETTE)):
        for j in range(i + 1, len(PALETTE)):
            d = float(np.linalg.norm(PAL[i] - PAL[j]))
            if d < 40:
                close.append((d, PALETTE[i][0], PALETTE[j][0]))
    if close:
        print("Palette colours that are close together (JPG noise could confuse them):")
        for d, a, b in sorted(close):
            print(f"  {a} <-> {b}: colour distance {d:.0f}")


def draw_check(rec, out_dir):
    img = cv2.imread(rec["img"])
    if img is None:
        return
    H, W = img.shape[:2]
    for c, x, y, w, h in rec["boxes"]:
        x0, y0, x1, y1 = int(x * W), int(y * H), int((x + w) * W), int((y + h) * H)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(img, CLASS_NAMES[c], (x0 + 3, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(str(Path(out_dir) / (Path(rec["img"]).stem + ".jpg")), img)


def write_extras(out_dir):
    base = Path(out_dir).resolve().parent
    (base / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n")
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    (base / "data_example.yaml").write_text(
        "# edit 'path' and the train/val/test folders to match your layout\n"
        f"path: {base.as_posix()}\ntrain: train/images\nval: validation/images\ntest: test/images\nnames:\n{names}\n")
    print(f"Also wrote {base / 'classes.txt'} and {base / 'data_example.yaml'}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", help="output folder for the .txt labels (not needed with --inspect)")
    ap.add_argument("--inspect", action="store_true", help="test on a few masks, write nothing, print a report")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (inspect default: 20)")
    ap.add_argument("--check", type=int, default=0, help="save N pictures with boxes drawn into box_check/")
    ap.add_argument("--workers", type=int, default=1, help="CPU processes (try 4)")
    ap.add_argument("--max-dist", type=float, default=60, help="colour distance above which a pixel counts as background")
    ap.add_argument("--smooth", type=int, default=3, help="majority-filter size in pixels (0 = off; try 5 if masks are speckled)")
    ap.add_argument("--min-area-frac", type=float, default=0.0005, help="ignore blobs smaller than this fraction of the image")
    ap.add_argument("--merge-gap", type=int, default=0, help="join pieces of a class closer than N pixels")
    ap.add_argument("--max-side", type=int, default=1000, help="shrink masks so the long side is at most this (0 = no shrink)")
    args = ap.parse_args()

    if not args.inspect and not args.out:
        raise SystemExit("Give --out (the labels folder), or use --inspect.")
    images = sorted(p for p in Path(args.images).rglob("*") if p.suffix.lower() in IMG_EXT)
    masks = build_mask_index(args.masks)
    jobs_src = [(p, masks[p.stem]) for p in images if p.stem in masks]
    missing = len(images) - len(jobs_src)
    if not jobs_src:
        raise SystemExit("No image has a matching mask. Check the folders and file names.")
    limit = args.limit or (20 if args.inspect else 0)
    if limit:
        jobs_src = jobs_src[:limit]
    print(f"{len(images)} images, {len(masks)} masks, {len(jobs_src)} to process, {missing} images without a mask")
    warn_close_colours()

    out_dir = None
    if not args.inspect:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"max_dist": args.max_dist, "smooth": args.smooth, "min_area_frac": args.min_area_frac,
           "merge_gap": args.merge_gap, "max_side": args.max_side}
    jobs = [(i, m, out_dir, cfg) for i, m in jobs_src]

    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            results = []
            for k, r in enumerate(ex.map(process_one, jobs, chunksize=16), 1):
                results.append(r)
                if k % 500 == 0:
                    print(f"  {k}/{len(jobs)}")
    else:
        results = []
        for k, job in enumerate(jobs, 1):
            results.append(process_one(job))
            if k % 500 == 0:
                print(f"  {k}/{len(jobs)}")

    per_class = Counter()
    empty = sum(1 for r in results if not r["boxes"])
    for r in results:
        for c, *_ in r["boxes"]:
            per_class[c] += 1
    far_avg = 100 * float(np.mean([r["far"] for r in results]))
    aspect_bad = sum(r["aspect_bad"] for r in results)

    print("\n=== REPORT ===")
    print(f"Images processed: {len(results)}   without any box: {empty}")
    print("Boxes per class:")
    for c in range(len(CLASS_NAMES)):
        print(f"  {c:2d} {CLASS_NAMES[c]:<11} {per_class.get(c, 0)}")
    print(f"Mask pixels far from every palette colour (JPG noise / edges): {far_avg:.1f}% on average")
    if far_avg > 8:
        print("  -> quite high: masks may be heavily compressed. Look at box_check pictures; try --smooth 5.")
    if aspect_bad:
        print(f"WARNING: {aspect_bad} masks have a different shape than their image; boxes may be misplaced.")

    if args.check:
        chk = Path("box_check")
        chk.mkdir(exist_ok=True)
        done = 0
        for r in results:
            if r["boxes"]:
                draw_check(r, chk)
                done += 1
                if done >= args.check:
                    break
        print(f"Saved {done} pictures with boxes in {chk}/ - look at them before training!")
    if out_dir is not None:
        print(f"Labels written to {out_dir.resolve()}")
        write_extras(out_dir)


if __name__ == "__main__":
    main()