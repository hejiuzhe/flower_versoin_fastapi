# Flower Vision

> 基于深度学习的花卉识别 API — 使用 MobileNetV3-Large 和 FastAPI 从图像中识别 102 种花卉。

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[English](README.md) | 中文

## 功能特性

- 识别 **102 种花卉**，基于 Oxford 102 Flowers 数据集训练
- **MobileNetV3-Large** 骨干网络，结合迁移学习，推理快速准确
- **FastAPI** 异步 REST API，自动生成 Swagger 文档
- 输出中文花卉名称（102 类标签映射）
- 同时支持 **GPU（CUDA）** 和 **CPU** 推理

## 技术栈

| 类别 | 技术 |
|---|---|
| 语言 | Python 3.12 |
| Web 框架 | FastAPI 0.136 |
| ASGI 服务器 | Uvicorn 0.47 |
| 深度学习 | PyTorch 2.8 + Torchvision 0.23 |
| 模型 | MobileNetV3-Large（迁移学习） |
| 图像处理 | Pillow 12.2 |
| 数据验证 | Pydantic 2.13 |
| 数据集 | Oxford 102 Flowers |

## 项目结构

```
flower_version/
├── app.py               # FastAPI 推理服务
├── train.py             # 模型训练脚本
├── data_clean.py         # 数据集加载工具
├── loaddata.py           # 数据集下载工具
├── flower_model.pth      # 训练好的模型权重（约 16.7 MB）
├── flower_labels.json    # 类别索引 → 花卉名称映射
├── requirements.txt      # Python 依赖
└── data/                 # 数据集目录（已 gitignore）
    └── flowers-102/      # Oxford 102 Flowers 数据集
```

## 快速开始

### 环境要求

- Python 3.12+
- CUDA 12.6（可选，用于 GPU 推理）

### 安装

```bash
git clone https://github.com/<your-username>/flower-vision.git
cd flower-vision

pip install -r requirements.txt
```

### 下载数据集

```bash
python loaddata.py
```

通过 torchvision 下载 Oxford 102 Flowers 数据集（8,189 张图片，102 个类别）。

### 训练模型

```bash
python train.py
```

训练配置：

| 参数 | 值 |
|---|---|
| 网络架构 | MobileNetV3-Large（ImageNet 预训练） |
| 数据增强 | RandomResizedCrop + RandomHorizontalFlip |
| 优化器 | AdamW（lr=1e-4, weight_decay=1e-4） |
| 学习率调度 | ReduceLROnPlateau（patience=2, factor=0.1） |
| 训练轮数 | 30 |
| 输出 | `flower_model.pth`（验证损失最优的模型） |

### 启动服务

```bash
python app.py
```

服务启动于 `http://127.0.0.1:8000`。访问 `http://127.0.0.1:8000/docs` 查看交互式 Swagger UI。

## API 参考

### `POST /predict`

根据图片 URL 预测花卉种类。

**请求**

```json
{
  "image_url": "https://example.com/flower.jpg"
}
```

**成功响应（200）**

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "flower_idx": 15,
    "flower_name": "欧金盏花",
    "confidence": 0.9876
  }
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `code` | `int` | 状态码 |
| `message` | `string` | 状态信息 |
| `data.flower_idx` | `int` | 预测类别索引（0–101） |
| `data.flower_name` | `string` | 花卉名称（中文） |
| `data.confidence` | `float` | 置信度（0–1） |

**错误响应（400）**

```json
{
  "code": 400,
  "message": "Failed to download image"
}
```

## 支持的花卉

模型可识别 Oxford 102 Flowers 数据集中的 102 种花卉。完整的类别到名称映射存储在 [`flower_labels.json`](flower_labels.json) 中。

## 许可证

本项目仅用于教育和研究目的。