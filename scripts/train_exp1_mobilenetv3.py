"""
训练 Exp1：YOLOv8n + MobileNetV3 主干网络。

本地或云端均可运行：
    python scripts/train_exp1_mobilenetv3.py
"""

from pathlib import Path
import os

# PyTorch 2.0 on Apple MPS does not implement hardsigmoid, used by MobileNetV3.
# Let unsupported operators fall back to CPU instead of aborting the run.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

# 兼容 PyTorch 2.6+ 加载旧格式权重。
original_load = torch.load


def patched_load(file, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return original_load(file, *args, **kwargs)


torch.load = patched_load

from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_CONFIG = PROJECT_DIR / "data" / "rdd2022.yaml"
MODEL_CONFIG = PROJECT_DIR / "models" / "yolov8n_mobilenetv3.yaml"


def main():
    model = YOLO(str(MODEL_CONFIG))

    if torch.cuda.is_available():
        device = 0
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # PyTorch 2.0.1 MPS lacks int64 cumsum used by Ultralytics' loss.
        print("检测到MPS；当前PyTorch版本存在算子兼容性问题，改用CPU保证训练稳定。")
        device = "cpu"
    else:
        device = "cpu"

    print(f"使用设备: {device}")
    print(f"数据配置: {DATA_CONFIG}")
    print(f"模型配置: {MODEL_CONFIG}")

    results = model.train(
        data=str(DATA_CONFIG),
        epochs=100,
        imgsz=640,
        batch=16,
        name="yolov8n_rdd2022_exp1_mobilenetv3",
        device=device,
        workers=0,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        plots=True,
    )

    print("训练完成！")
    print(f"最佳模型保存位置：{results.save_dir}")


if __name__ == "__main__":
    main()
