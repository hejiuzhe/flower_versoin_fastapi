import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def main():
    # ==========================================
    # 1. 硬件与路径配置（纯本地）
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的训练设备: {device}")

    # 替换为你本地的绝对路径（r"" 可以防止 Windows 下反斜杠转义报错）
    LOCAL_DATA_PATH = r"D:\flower_version\data"

    if not os.path.exists(LOCAL_DATA_PATH):
        print(f"❌ 错误：在本地找不到路径 {LOCAL_DATA_PATH}，请检查文件夹是否存在！")
        return

    # ==========================================
    # 2. 本地数据处理与流加载（Data Pipeline）
    # ==========================================
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # 训练集：加入数据增强防止小样本过拟合
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # 验证集：不做随机变换，只规整尺寸
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    print("正在直接读取本地硬盘数据并进行矩阵转换...")
    # 提示：由于Oxford官方把大样本放到了 test 划分，我们用 test 划分做训练，train 划分做验证
    train_dataset = datasets.Flowers102(root=LOCAL_DATA_PATH, split="test", download=True, transform=train_transforms)
    val_dataset = datasets.Flowers102(root=LOCAL_DATA_PATH, split="train", download=True, transform=val_transforms)

    # 打包成 batch 喂给模型
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"✅ 数据加载成功！训练集: {len(train_dataset)}张, 验证集: {len(val_dataset)}张")

    # ==========================================
    # 3. 模型初始化与核心配置
    # ==========================================
    print("正在构建 MobileNetV3-Large 迁移学习网络...")
    # 载入带预训练权重的经典轻量化模型
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

    # 改写最后一层全连接，将输出维度修改为 102 类
    num_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_features, 102)
    model = model.to(device)

    # 损失函数、优化器与动态学习率策略
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1)

    # ==========================================
    # 4. 完整的模型训练与验证循环（核心逻辑）
    # ==========================================
    num_epochs = 30  # 预训练模型微调，学生项目跑 10 遍左右基本就收敛了
    print(f"🔥 配置完毕，开始执行 {num_epochs} 轮训练...")

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # --- 训练阶段 ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = 100. * correct_train / total_train

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = 100. * correct_val / total_val

        # 更新学习率策略（盯着验证集的 Loss）
        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # 打印当前 Epoch 的战果
        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"| Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% "
              f"| Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}% "
              f"| LR: {current_lr}")

        # 如果这一轮的表现是最好的，就保存它的权重
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            # 只保存模型的权重参数，这是做 API 接口时最规范的载入方式
            torch.save(model.state_dict(), "flower_model.pth")
            print("👉 验证集表现提升，已更新并保存模型为 flower_model.pth")

    print("\n🎉 训练全部结束！最终的最优模型权重已妥善保存在当前目录下的 flower_model.pth 中。")


if __name__ == "__main__":
    main()