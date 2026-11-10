"""
preview_corruptions.py
----------------------
Makes ONE picture showing every corruption at severity 1, 2 and 3 for a single image,
so you can check by eye that the effects look realistic (and are not too strong).

Usage:
    python preview_corruptions.py --image sample.jpg --mask sample_mask.png --out preview.png
    python preview_corruptions.py --image sample.jpg --out preview.png      # no mask: occlusion goes anywhere

Rule of thumb: if you cannot recognise the glass object yourself at severity 3,
the effect is too strong and the results will not mean much.
"""

import argparse

import cv2
import numpy as np

from corruptions import CORRUPTIONS, SEVERITIES, apply

CELL = 256  # size of each small picture


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (CELL, 22), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", help="optional mask (needed to place occlusion on the object)")
    ap.add_argument("--out", default="preview.png")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Cannot read {args.image}")
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE) if args.mask else None
    if mask is not None and mask.shape != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    def small(x):
        return cv2.resize(x, (CELL, CELL))

    rows = []
    for name in CORRUPTIONS:
        cells = [label(small(img), "original")]
        for sev in SEVERITIES:
            rng = np.random.default_rng(args.seed)
            out, info = apply(name, img, sev, rng, mask)
            text = f"{name} s{sev}"
            if "coverage" in info:
                text += f" ({int(info['coverage'] * 100)}% hidden)"
            cells.append(label(small(out), text))
        rows.append(np.hstack(cells))
    cv2.imwrite(args.out, np.vstack(rows))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()