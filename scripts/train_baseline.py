"""
训练 YOLOv8n baseline
论文：MCA-YOLO-A 复现第一步
数据集：RDD2022 China_MotorBike (2477张，4类)
"""

# 修复 PyTorch 2.6+ weights_only 兼容问题
import torch
original_load = torch.load
def _patched_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return original_load(f, *args, **kwargs)
torch.load = _patched_load

from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolov8n.pt")

# 开始训练
results = model.train(
    data="/Users/zhangkai/Documents/mca-yolo-reproduce/data/rdd2022.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="yolov8n_rdd2022_baseline",
    device="cpu",
)

print("训练完成！")
print(f"最佳模型保存位置：{results.save_dir}")
