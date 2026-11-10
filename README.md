# Transparent Object Detection (YOLO11)

Detects transparent objects in images, videos and live webcam using [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics). Includes scripts for training and for inference on images, folders, videos and a live camera.

## Setup

```bash
git clone https://github.com/aayushnemade-2005/Transparent_object_detection.git
cd Transparent_object_detection

python -m venv yolo-env
yolo-env\Scripts\activate        # Windows

# Install PyTorch with CUDA first (get the command from https://pytorch.org/get-started/locally/)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX
python -m pip install -e .
```

Use Python 3.10-3.12 (64-bit). Check the GPU with `python -c "import torch; print(torch.cuda.is_available())"`, which should print `True`.

## Dataset

YOLO format with class IDs starting at **0**:

```
datasets/transparent/
├── images/{train,val}
└── labels/{train,val}
```

Set the paths and class names in `transparent_obj_det.yaml`.

## Train

```bash
python train.py --data transparent_obj_det.yaml --model yolo11s.pt --epochs 100 --imgsz 640 --batch 16 --device 0

# Resume an interrupted run
python train.py --resume runs/transparent/exp1/weights/last.pt
```

The best model is saved as `weights/best.pt` in the run folder.

## Inference

```bash
python inference.py --source test.jpg            # image
python inference.py --source path/to/images      # folder
python inference.py --source clip.mp4            # video
python inference.py --source 1                   # live webcam (q = quit, f = fullscreen)
```

Useful options: `--weights best.pt`, `--conf 0.6` (confidence threshold), `--half` (faster on GPU), `--flip`, `--winsize 1280x720`, `--save` (record live output).
Classes to ignore are listed in `EXCLUDE_CLASSES` at the top of `inference.py`.

## Model weights

Download `best.pt` from the [Releases](../../releases) page.

## License and credits

Built on [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) and licensed under AGPL-3.0.