"""Train Exp1: YOLOv8n with a 160x160 P2 small-object detection head."""

from pathlib import Path

import torch

# Keep compatibility with checkpoints produced by newer PyTorch versions.
original_load = torch.load


def patched_load(file, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return original_load(file, *args, **kwargs)


torch.load = patched_load

from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_CONFIG = PROJECT_DIR / "data" / "rdd2022.yaml"
MODEL_CONFIG = PROJECT_DIR / "models" / "yolov8n_smallhead.yaml"
PRETRAINED_WEIGHTS = PROJECT_DIR / "yolov8n.pt"


def main():
    model = YOLO(str(MODEL_CONFIG))
    if PRETRAINED_WEIGHTS.exists():
        model.load(str(PRETRAINED_WEIGHTS))

    results = model.train(
        data=str(DATA_CONFIG),
        epochs=100,
        imgsz=640,
        batch=16,
        name="yolov8n_rdd2022_exp1_smallhead",
        device=0 if torch.cuda.is_available() else "cpu",
        plots=True,
    )

    print(f"训练完成！最佳模型保存在: {results.save_dir}")


if __name__ == "__main__":
    main()
