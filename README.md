# Flower Vision

> Deep learning-based flower recognition API — identify 102 flower species from images using MobileNetV3-Large and FastAPI.

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

English | [中文](README-zh.md)

## Features

- Recognizes **102 flower species** trained on the Oxford 102 Flowers dataset
- **MobileNetV3-Large** backbone with transfer learning for fast, accurate inference
- **FastAPI** async REST API with automatic Swagger docs
- Chinese flower name output (102-class label mapping)
- Supports both **GPU (CUDA)** and **CPU** inference

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.12 |
| Web Framework | FastAPI 0.136 |
| ASGI Server | Uvicorn 0.47 |
| Deep Learning | PyTorch 2.8 + Torchvision 0.23 |
| Model | MobileNetV3-Large (transfer learning) |
| Image Processing | Pillow 12.2 |
| Validation | Pydantic 2.13 |
| Dataset | Oxford 102 Flowers |

## Project Structure

```
flower_version/
├── app.py               # FastAPI inference server
├── train.py             # Model training script
├── data_clean.py         # Dataset loader utility
├── loaddata.py           # Dataset download utility
├── flower_model.pth      # Trained model weights (~16.7 MB)
├── flower_labels.json    # Class index → flower name mapping
├── requirements.txt      # Python dependencies
└── data/                 # Dataset directory (gitignored)
    └── flowers-102/      # Oxford 102 Flowers dataset
```

## Quick Start

### Prerequisites

- Python 3.12+
- CUDA 12.6 (optional, for GPU inference)

### Installation

```bash
git clone https://github.com/<your-username>/flower-vision.git
cd flower-vision

pip install -r requirements.txt
```

### Download the Dataset

```bash
python loaddata.py
```

This downloads the Oxford 102 Flowers dataset (8,189 images, 102 classes) via torchvision.

### Train the Model

```bash
python train.py
```

Training configuration:

| Parameter | Value |
|---|---|
| Architecture | MobileNetV3-Large (ImageNet pretrained) |
| Augmentation | RandomResizedCrop + RandomHorizontalFlip |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (patience=2, factor=0.1) |
| Epochs | 30 |
| Output | `flower_model.pth` (best val loss) |

### Start the Server

```bash
python app.py
```

The server starts at `http://127.0.0.1:8000`. Visit `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

## API Reference

### `POST /predict`

Predict the flower species from an image URL.

**Request**

```json
{
  "image_url": "https://example.com/flower.jpg"
}
```

**Response (200)**

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

| Field | Type | Description |
|---|---|---|
| `code` | `int` | Status code |
| `message` | `string` | Status message |
| `data.flower_idx` | `int` | Predicted class index (0–101) |
| `data.flower_name` | `string` | Flower name (Chinese) |
| `data.confidence` | `float` | Confidence score (0–1) |

**Error Response (400)**

```json
{
  "code": 400,
  "message": "Failed to download image"
}
```

## Supported Flowers

The model recognizes 102 flower species from the Oxford 102 Flowers dataset. The full class-to-name mapping is stored in [`flower_labels.json`](flower_labels.json).

## License

This project is for educational and research purposes.