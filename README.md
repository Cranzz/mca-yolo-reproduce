# MCA-YOLO-A 复现项目

> 目标：复现论文《基于改进 YOLOv8n 的道路裂缝检测轻量化模型》

## 项目结构

```
mca-yolo-reproduce/
├── data/          # 数据集
│   └── RDD2022/   # 下载后放这里
├── models/        # 模型定义（MobileNetV3 + CA 注意力）
├── scripts/       # 训练、推理、评估脚本
├── configs/       # 训练配置文件
├── runs/          # 训练日志和结果
└── README.md
```

## 阶段一：环境搭建 + 跑通 baseline

- [ ] 装环境：pytorch + ultralytics
- [ ] 下载 RDD2022 数据集
- [ ] 转成 YOLO 格式
- [ ] 跑通 YOLOv8n 官方模型训练
- [ ] 记录 baseline 指标

## 阶段二：复现改进

- [ ] 实现 MobileNetV3 替换主干
- [ ] 插入 CA 注意力模块
- [ ] 换成 Alpha-IOU 损失函数
- [ ] 增加 160x160 小目标检测层
- [ ] 训练并记录指标

## 阶段三：对比分析

- [ ] 对比 baseline vs 改进后 mAP
- [ ] 对比参数量和推理速度
- [ ] 可视化检测结果对比

## 参考

- 论文：04-技术学习/02-MCA-YOLO复现/MCA-YOLO-A_论文.pdf
- 笔记：04-技术学习/03-论文笔记/MCA-YOLO-A精读笔记.md
- YOLOv8 官方文档：https://docs.ultralytics.com
