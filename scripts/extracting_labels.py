import cv2
import os
import numpy as np

# ==========================================
# 1. DEFINE DATASET PATHS
# ==========================================

IMAGE_DIR = "dataset/images"
MASK_DIR = "dataset/masks"
LABEL_DIR = "dataset/labels"

os.makedirs(LABEL_DIR, exist_ok=True)

# ==========================================
# 2. PROCESS MASKS AND EXTRACT BOUNDING BOXES
# ==========================================

for mask_file in os.listdir(MASK_DIR):

    mask_path = os.path.join(MASK_DIR, mask_file)

    # Read mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print("Unable to read:", mask_file)
        continue

    # Convert mask into binary format
    _, binary = cv2.threshold(
        mask, 0, 255, cv2.THRESH_BINARY
    )

    # Find connected components
    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(binary, connectivity=8)

    height, width = binary.shape

    # Corresponding image filename
    image_name = os.path.splitext(mask_file)[0]

    # YOLO label file
    label_path = os.path.join(
        LABEL_DIR, image_name + ".txt"
    )

    with open(label_path, "w") as f:

        # Skip background (component 0)
        for i in range(1, num_labels):

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]

            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            area = stats[i, cv2.CC_STAT_AREA]

            # Ignore very small noise
            if area < 20:
                continue

            # Convert to YOLO format
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height

            box_width = w / width
            box_height = h / height

            # Single class: transparent object
            class_id = 0

            f.write(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}\n"
            )

    print("Labels generated:", image_name)

print("\nAll YOLO labels generated successfully!")