import os
import io
import json
import torch
import torch.nn as nn
import requests
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
import torchvision.models as models
import torchvision.transforms as transforms

# 初始化 FastAPI 实例
app = FastAPI(
    title="花卉视觉识别模型 API ",
    description="基于本地大模型识别JSON字典进行动态查表，0代码硬编码",
    version="2.1.1"
)

# ==========================================
# 1. 核心初始化：服务启动时，一次性载入模型与 JSON 标签配置
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 FastAPI 推理引擎正在初始化，使用设备: {device}")

# 🌟【动态解耦】：直接从你刚刚自动清洗生成的本地唯一真 JSON 中载入映射关系
JSON_PATH = "flower_labels.json"
if os.path.exists(JSON_PATH):
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        # 注意：JSON 解析出来后的键（Key）全部是字符串，如 "0", "76", "101"
        FLOWER_CLASSES_FROM_JSON = json.load(f)
    print(f"📖 成功动态加载大模型识花标签库: {JSON_PATH}，共计 {len(FLOWER_CLASSES_FROM_JSON)} 个分类。")
else:
    raise FileNotFoundError(f"❌ 严重错误：未在当前目录下找到 {JSON_PATH}！请确保你生成的 JSON 文件在同一个文件夹下。")

# 构建与训练完全一致的 MobileNetV3-Large 架构
model = models.mobilenet_v3_large(weights=None)
num_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_features, 102)

# 载入核心权重
MODEL_PATH = "flower_model.pth"
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"✅ 成功载入本地权重文件: {MODEL_PATH}")
else:
    print(f"❌ 错误：在当前目录下未找到 {MODEL_PATH}，请确保权重文件在同级目录！")

model = model.to(device)
model.eval()

# ==========================================
# 2. 定义图像预处理流 (与你的训练验证集完美对齐)
# ==========================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

infer_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


# ==========================================
# 3. 定义请求的 JSON 数据格式
# ==========================================
class ImageUrlRequest(BaseModel):
    image_url: HttpUrl


# ==========================================
# 4. 编写预测接口
# ==========================================
@app.post("/predict", summary="传入包含图片 URL 的 JSON 进行精准花卉分类识别")
async def predict(request_data: ImageUrlRequest):
    url_str = str(request_data.image_url)

    try:
        # 1. 抓取远程 OBS 或网络图片
        response = requests.get(url_str, timeout=5)
        if response.status_code != 200:
            raise HTTPException(status_code=400,
                                detail=f"图片下载失败，网络图片服务器返回状态码: {response.status_code}")

        # 2. 将图片载入内存并转换为标准的 RGB 模式
        image_bytes = response.content
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 3. 执行严格对齐的图像特征预处理并升维增加 Batch 维度
        input_tensor = infer_transforms(image).unsqueeze(0).to(device)

        # 4. 模型推理计算
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)

        class_id = predicted_idx.item()

        # 🌟【查表核心改动】：因为 JSON 的 Key 读出来是字符串，所以这里必须转成 str(class_id)
        # 此时查出来的 76，就会直接吐出大模型认出的“非洲菊”，绝不发生半个索引的错位！
        flower_name = FLOWER_CLASSES_FROM_JSON.get(str(class_id), f"未知花卉_数字标签:{class_id}")

        return {
            "code": 200,
            "message": "success",
            "data": {
                "flower_idx": class_id,
                "flower_name": flower_name,
                "confidence": round(confidence.item(), 4)
            }
        }

    except requests.exceptions.RequestException as net_err:
        return JSONResponse(
            status_code=400,
            content={"code": 400, "message": f"接口无法抓取该图片 URL，请检查网络。详情: {str(net_err)}"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"code": 500, "message": f"视觉模型内部推理遇到障碍: {str(e)}"}
        )


if __name__ == "__main__":
    import uvicorn

    # 启动服务
    uvicorn.run(app, host="127.0.0.1", port=8000)