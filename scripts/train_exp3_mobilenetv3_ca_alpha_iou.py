"""
训练 Exp3：YOLOv8n + MobileNetV3 + CA + Alpha-IoU。

本地或云端均可运行：
    python scripts/train_exp3_mobilenetv3_ca_alpha_iou.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

original_load = torch.load


def patched_load(file, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return original_load(file, *args, **kwargs)


torch.load = patched_load

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from models.modules import (  # noqa: E402
    AlphaIoUDetectionLoss,
    CoordinateAttention,
    MobileNetV3CA,
    add_alpha_iou_loss,
)
from ultralytics import YOLO  # noqa: E402
from ultralytics.nn import tasks  # noqa: E402


tasks.MobileNetV3CA = MobileNetV3CA
tasks.TorchVision = MobileNetV3CA

DATA_CONFIG = PROJECT_DIR / "data" / "rdd2022.yaml"
MODEL_CONFIG = PROJECT_DIR / "models" / "yolov8n_mobilenetv3_ca.yaml"
ALPHA = 3.0


def main():
    model = YOLO(str(MODEL_CONFIG))
    ca_count = sum(
        isinstance(module, CoordinateAttention) for module in model.model.modules()
    )
    if ca_count == 0:
        raise RuntimeError("未在MobileNetV3中注册任何CA模块。")

    add_alpha_iou_loss(model, alpha=ALPHA)

    if torch.cuda.is_available():
        device = 0
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("检测到MPS；当前PyTorch版本存在算子兼容性问题，改用CPU保证训练稳定。")
        device = "cpu"
    else:
        device = "cpu"

    print(f"CA模块数量: {ca_count}")
    print(f"Alpha-IoU alpha: {ALPHA:g}")
    print(f"使用设备: {device}")
    print(f"数据配置: {DATA_CONFIG}")
    print(f"模型配置: {MODEL_CONFIG}")

    results = model.train(
        data=str(DATA_CONFIG),
        epochs=100,
        imgsz=640,
        batch=16,
        name="yolov8n_rdd2022_exp3_mobilenetv3_ca_alpha_iou",
        device=device,
        workers=0,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        plots=True,
    )

    if not isinstance(model.model.criterion, AlphaIoUDetectionLoss):
        raise RuntimeError("训练过程中未启用 Alpha-IoU 损失。")

    print("训练完成！")
    print(f"最佳模型保存位置：{results.save_dir}")


if __name__ == "__main__":
    main()

