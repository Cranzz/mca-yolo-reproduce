"""Train Exp1: YOLOv8n with only the MobileNetV3 backbone replacement."""

from pathlib import Path

import torch

from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_CONFIG = PROJECT_DIR / "data" / "rdd2022.yaml"
MODEL_CONFIG = PROJECT_DIR / "models" / "yolov8n_mobilenetv3.yaml"


def main():
    model = YOLO(str(MODEL_CONFIG))
    results = model.train(
        data=str(DATA_CONFIG),
        epochs=100,
        imgsz=640,
        batch=16,
        name="yolov8n_rdd2022_exp1_mobilenetv3",
        device=0 if torch.cuda.is_available() else "cpu",
        plots=True,
    )
    print(f"训练完成！最佳模型保存在: {results.save_dir}")


if __name__ == "__main__":
    main()
