import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# 定义数据集存放路径
data_dir = "./oxford_102"

# 基础图像预处理（可以先只转为张量）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

print("正在下载并加载 Oxford 102 训练集...")
# split 可以是 'train' (训练), 'val' (验证), 或 'test' (测试)
train_dataset = datasets.Flowers102(
    root=data_dir,
    split='train',
    download=True,
    transform=transform
)

print(f"下载完成！数据集已保存至: {os.path.abspath(data_dir)}")
print(f"训练集样本数量: {len(train_dataset)}")