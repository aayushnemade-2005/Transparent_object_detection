"""
parquet_to_folders.py
---------------------
Turns a Hugging Face dataset saved as .parquet files into normal folders:

    out/train/images/xxx.jpg        out/train/masks/xxx.png
    out/validation/images/...       out/validation/masks/...
    out/test/images/...             out/test/masks/...

Masks are saved from their ORIGINAL bytes (never re-encoded), so the class numbers
inside the mask stay exactly as the dataset authors stored them.

Install once:   pip install pyarrow pillow numpy

STEP 1 - look inside the files (columns, image sizes, mask values):
    python parquet_to_folders.py --parquet-dir C:\\data\\trans10k_parquet --inspect

STEP 2 - extract (add --limit 20 first to test on a few images per split):
    python parquet_to_folders.py --parquet-dir C:\\data\\trans10k_parquet --out Trans10K_v2

If it cannot guess which column is the image and which is the mask, pass them:
    --image-col image --mask-col mask

The split (train / validation / test) is guessed from the file or folder name.
Use --split train to force one split for every file.

Then continue with:
    python masks_to_yolo.py --images Trans10K_v2/train/images --masks Trans10K_v2/train/masks --inspect
"""

import argparse
import io
import json
import re
from collections import Counter
from pathlib import Path


# ------------------------------------------------------------------ pyarrow
def _pyarrow():
    try:
        import pyarrow
        import pyarrow.parquet
        return pyarrow
    except ImportError:
        raise SystemExit("pyarrow is missing. Install it with:  pip install pyarrow")


def open_parquet(path):
    pa = _pyarrow()
    return pa.parquet.ParquetFile(str(path))


def describe_schema(pf):
    """List of (name, kind, subfields). kind: 'image' (struct with bytes), 'binary', 'other'."""
    pa = _pyarrow()
    out = []
    for field in pf.schema_arrow:
        t = field.type
        if pa.types.is_struct(t):
            subs = [t.field(i).name for i in range(t.num_fields)]
            kind = "image" if "bytes" in subs else "other"
        elif pa.types.is_binary(t) or pa.types.is_large_binary(t):
            kind, subs = "binary", []
        else:
            kind, subs = "other", []
        out.append((field.name, kind, subs))
    return out


# ------------------------------------------------------------------ helpers
MASK_WORDS = re.compile(r"(mask|label|annot|seg|gt)", re.I)


def guess_split(path):
    for part in reversed(Path(path).parts):
        low = part.lower()
        if "validation" in low or re.search(r"(^|[^a-z])val(id)?([^a-z]|$)", low):
            return "validation"
        if "test" in low:
            return "test"
        if "train" in low:
            return "train"
    return None


def choose_columns(described, image_col=None, mask_col=None):
    names = [n for n, kind, _ in described if kind in ("image", "binary")]
    if image_col and mask_col:
        return image_col, mask_col
    if image_col:
        rest = [n for n in names if n != image_col]
        if len(rest) == 1:
            return image_col, rest[0]
    if mask_col:
        rest = [n for n in names if n != mask_col]
        if len(rest) == 1:
            return rest[0], mask_col
    masks = [n for n in names if MASK_WORDS.search(n)]
    images = [n for n in names if n not in masks]
    if len(images) == 1 and len(masks) == 1:
        return images[0], masks[0]
    raise SystemExit(
        f"Could not tell the image column from the mask column. Image-like columns: {names}. "
        "Run with --inspect, then pass --image-col NAME --mask-col NAME.")


def ext_from_bytes(b):
    if b[:4] == b"\x89PNG":
        return ".png"
    if b[:2] == b"\xff\xd8":
        return ".jpg"
    if b[:2] == b"BM":
        return ".bmp"
    return ".png"


def get_bytes(value):
    """Return (bytes, original_path_or_None) for an image cell."""
    if value is None:
        return None, None
    if isinstance(value, dict):
        b, p = value.get("bytes"), value.get("path")
        if b is None and p and Path(p).exists():
            b = Path(p).read_bytes()
        return b, p
    if isinstance(value, (bytes, bytearray)):
        return bytes(value), None
    # fallback: a nested list / array of numbers -> save as a grey PNG
    import numpy as np
    from PIL import Image
    arr = np.asarray(value)
    if arr.ndim == 2:
        buf = io.BytesIO()
        Image.fromarray(arr.astype("uint8")).save(buf, format="PNG")
        return buf.getvalue(), None
    return None, None


def mask_values(b, limit=20):
    import numpy as np
    from PIL import Image
    m = np.array(Image.open(io.BytesIO(b)))
    return [int(v) for v in np.unique(m)][:limit], m.shape, Image.open(io.BytesIO(b)).mode


# ------------------------------------------------------------------ inspect
def inspect(files, image_col, mask_col):
    from PIL import Image
    print(f"Found {len(files)} parquet file(s):")
    rows_by_split = Counter()
    for f in files:
        pf = open_parquet(f)
        n = pf.metadata.num_rows
        split = guess_split(f)
        rows_by_split[split] += n
        print(f"  {f.name}  rows={n}  split guess={split}")
    print("\nRows per split guess:", dict(rows_by_split))

    pf = open_parquet(files[0])
    desc = describe_schema(pf)
    print(f"\nColumns in {files[0].name}:")
    for name, kind, subs in desc:
        extra = f"  fields={subs}" if subs else ""
        print(f"  {name}: {kind}{extra}")
    try:
        meta = pf.schema_arrow.metadata or {}
        if b"huggingface" in meta:
            feats = json.loads(meta[b"huggingface"]).get("info", {}).get("features", {})
            print("\nHugging Face feature description:", json.dumps(feats)[:600])
    except Exception:
        pass

    try:
        img_c, msk_c = choose_columns(desc, image_col, mask_col)
        print(f"\nColumns I would use -> image: '{img_c}', mask: '{msk_c}'")
    except SystemExit as e:
        print("\n" + str(e))
        return
    batch = next(pf.iter_batches(batch_size=3, columns=[img_c, msk_c]))
    for i, row in enumerate(batch.to_pylist()):
        ib, ip = get_bytes(row[img_c])
        mb, mp = get_bytes(row[msk_c])
        if ib is None or mb is None:
            print(f"row {i}: missing image or mask data")
            continue
        im = Image.open(io.BytesIO(ib))
        vals, shape, mode = mask_values(mb)
        print(f"row {i}: image {im.size} {im.mode} path={ip} | mask shape={shape} mode={mode} "
              f"values={vals} ({ext_from_bytes(mb)})")
    print("\nFor Trans10K the mask should be 1-channel (mode L or P); value 0 = background.")


# ------------------------------------------------------------------ extract
def extract(files, out_root, image_col, mask_col, forced_split, limit, batch_size):
    counters = Counter()
    seen = {}
    values_seen = Counter()
    warned_jpeg = False
    for f in files:
        split = forced_split or guess_split(f)
        if split is None:
            print(f"WARNING: cannot guess the split of {f.name}; using 'train'. Use --split to set it.")
            split = "train"
        pf = open_parquet(f)
        img_c, msk_c = choose_columns(describe_schema(pf), image_col, mask_col)
        img_dir = Path(out_root) / split / "images"
        msk_dir = Path(out_root) / split / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        msk_dir.mkdir(parents=True, exist_ok=True)
        used = seen.setdefault(split, set())
        print(f"{f.name} -> {split}  (image='{img_c}', mask='{msk_c}')")

        for batch in pf.iter_batches(batch_size=batch_size, columns=[img_c, msk_c]):
            for row in batch.to_pylist():
                if limit and counters[split] >= limit:
                    break
                ib, ip = get_bytes(row[img_c])
                mb, _ = get_bytes(row[msk_c])
                if ib is None or mb is None:
                    print("  skipped a row with missing data")
                    continue
                idx = counters[split]
                stem = Path(ip).stem if ip else f"{split}_{idx:06d}"
                if stem in used:
                    stem = f"{stem}_{idx:06d}"
                used.add(stem)
                mext = ext_from_bytes(mb)
                if mext == ".jpg" and not warned_jpeg:
                    print("  WARNING: masks are stored as JPEG; class numbers may be damaged.")
                    warned_jpeg = True
                (img_dir / f"{stem}{ext_from_bytes(ib)}").write_bytes(ib)
                (msk_dir / f"{stem}{mext}").write_bytes(mb)
                if counters[split] < 50:
                    try:
                        for v in mask_values(mb, limit=100)[0]:
                            values_seen[v] += 1
                    except Exception:
                        pass
                counters[split] += 1
            if limit and counters[split] >= limit:
                break

    print("\nDone. Images written per split:", dict(counters))
    if values_seen:
        print("Mask values seen in the first 50 masks of each run:", sorted(values_seen))
    print(f"Output folder: {Path(out_root).resolve()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet-dir", required=True, help="folder containing the .parquet files (searched recursively)")
    ap.add_argument("--out", default="Trans10K_extracted", help="output folder")
    ap.add_argument("--inspect", action="store_true", help="only show the columns and a few rows, then stop")
    ap.add_argument("--image-col")
    ap.add_argument("--mask-col")
    ap.add_argument("--split", choices=["train", "validation", "test"], help="force the split for all files")
    ap.add_argument("--limit", type=int, default=0, help="extract only N images per split (for a quick test)")
    ap.add_argument("--batch-size", type=int, default=32, help="rows read at a time (lower it if RAM is low)")
    args = ap.parse_args()

    files = sorted(Path(args.parquet_dir).rglob("*.parquet"))
    if not files:
        raise SystemExit(f"No .parquet files found under {args.parquet_dir}")
    if args.inspect:
        inspect(files, args.image_col, args.mask_col)
    else:
        extract(files, args.out, args.image_col, args.mask_col, args.split, args.limit, args.batch_size)


if __name__ == "__main__":
    main()