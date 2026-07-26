# MCA-YOLO-A 复现项目

> 目标：复现论文《基于改进 YOLOv8n 的道路裂缝检测轻量化模型》
>
> 作者：朱佳慧、刘艺、张登银（南京邮电大学物联网学院）
> 期刊：数据采集与处理，2025，CSCD+北大核心

## 项目结构

```
mca-yolo-reproduce/
├── data/                    # 数据集（RDD2022 China_MotorBike + yolo_format）
├── models/                  # 模型定义 yaml
│   ├── yolov8n_baseline.yaml      # baseline（原始 YOLOv8n）
│   ├── yolov8n_smallhead.yaml     # + 小目标检测层 (P2, 160x160)
│   ├── yolov8n_alpha_iou.yaml     # + P2 + Alpha-IOU
│   └── modules/                   # 自定义模块（TODO）
├── scripts/                 # 训练脚本
│   ├── train_baseline.py          # 本地 baseline 训练
│   ├── train_exp1_mobilenetv3.py  # Exp1：MobileNetV3主干训练（本地/云端通用）
│   ├── train_exp1_mobilenetv3_colab.ipynb # Exp1：Colab 云端训练
│   ├── train_mca_yolo_a.py        # 完整 MCA-YOLO-A 复现
│   ├── train_colab.ipynb          # Colab GPU 训练 Notebook
│   └── convert_xml_to_yolo.py     # VOC XML → YOLO txt 转换
├── configs/                 # 实验配置文件
│   ├── baseline.yaml
│   ├── exp1_mobilenetv3.yaml
│   ├── exp1_smallhead.yaml
│   ├── exp2_alpha_iou.yaml
│   └── exp3_mca_yolo_a.yaml
├── runs/                    # 训练结果（自动生成）
├── data/rdd2022.yaml        # 数据集配置
├── .gitignore
└── README.md
```

## 数据集

本地复现使用 `China_MotorBike/train` 中具有 XML 标注的 1977 张图片，并按固定随机种子 42 划分为训练集 1383 张、验证集 296 张、测试集 298 张。原始 `test/images` 中的约500张图片没有XML标注，因此不纳入监督训练和评估。

| 数据集 | 说明 | 下载源 |
|---|---|---|
| RDD2022 China_MotorBike | 论文主数据集，2477 张中国道路裂缝图 | [官方 S3 链接](https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_China_MotorBike.zip) |

## 论文改进点

1. **MobileNetV3 替换主干**：轻量化，降低参数量
2. **CA 注意力模块**：补偿空间位置信息，提升小目标检测
3. **Alpha-IOU 损失函数**：加速收敛，提升定位精度
4. **P2 小目标检测层 (160x160)**：提升细小裂缝识别

## 实验进度

- [x] 环境搭建 + 数据准备
- [x] baseline（YOLOv8n）训练完成
- [ ] Exp1：+ 小目标检测层
- [ ] Exp2：+ 小目标检测层 + Alpha-IOU
- [ ] Exp3：+ MobileNetV3 + CA + P2 + Alpha-IOU（完整 MCA-YOLO-A）
- [ ] 多模型对比分析

## 实验结果

| 模型 | mAP50 | 参数量 | FPS |
|---|---|---|---|
| YOLOv8n (baseline) | TBD | — | — |
| + 小目标层 | TBD | — | — |
| + Alpha-IOU | TBD | — | — |
| **MCA-YOLO-A (论文)** | **0.930** | **6.0M** | **95** |
| MCA-YOLO-A (复现) | TBD | TBD | TBD |

## 复现参考

- 论文：`MCA-YOLO-A_论文.pdf`（存于 Obsidian 学习库）
- 精读笔记：`04-技术学习/03-论文笔记/MCA-YOLO-A精读笔记.md`

## Colab 训练

在 [Google Colab](https://colab.research.google.com) 上传 `scripts/train_colab.ipynb`，切换 T4 GPU 后依次运行所有单元格即可。代码和数据会自动从 GitHub 拉取。

## Exp1 训练

在 Colab 中打开 `scripts/train_exp1_mobilenetv3_colab.ipynb`，训练仅替换 MobileNetV3 主干的 Exp1。脚本使用当前 `data/rdd2022.yaml` 中的1977张有标注图片，默认训练100轮、输入尺寸640、batch size为16。
