import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 1. 明确指定你电脑上的绝对路径
# 使用 r"" 可以防止 Windows 系统下的反斜杠 \ 带来转义字符的报错
LOCAL_DATA_PATH = r"D:\flower_version\data"

# 检查一下路径，确保文件夹确实存在
if not os.path.exists(LOCAL_DATA_PATH):
    print(f"❌ 警告：在本地找不到路径 {LOCAL_DATA_PATH}，请检查盘符和文件夹名字！")
else:
    print(f"✅ 成功定位本地数据目录: {LOCAL_DATA_PATH}")

# 2. 定义常规的图像处理流
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

print("正在直接读取本地硬盘数据并进行矩阵转换...")

# 3. 传入绝对路径，PyTorch 会直接在 D 盘你的目录下读取 102flowers.tgz 等文件
# 提示：download=True 留着是 PyTorch 的语法要求（用于激活解压和读取逻辑），检测到本地有文件它绝不联网
train_dataset = datasets.Flowers102(
    root=LOCAL_DATA_PATH,
    split="test",
    download=True,
    transform=train_transforms
)

# 4. 包装成 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print("\n--- 本地硬盘数据加载成功 ---")
print(f"已直接从 D 盘加载了 {len(train_dataset)} 张本地图片，并转换为了模型可用的张量流。")