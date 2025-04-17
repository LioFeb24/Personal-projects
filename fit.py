import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
from PIL import Image
import random


if __name__ == '__main__':
    # 数据路径
    data_path = './data'
    # 判断是否有 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 数据增强 transform
    augment_transform = transforms.Compose([
        transforms.Resize((140, 140)),  # 放大
        transforms.RandomCrop((128, 128)),  # 再随机裁剪成目标尺寸
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色扰动
        transforms.RandomRotation(degrees=15),  # 随机旋转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    # 2. 固定预处理（可选用于验证集）
    static_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    # 原始数据加载（ImageFolder 只获取路径）
    raw_dataset = datasets.ImageFolder(root=data_path)

    # 提取图片和标签（增强 N 次）
    all_images = []
    all_labels = []
    augment_times = 2  # 每张图增强几次（可调）

    for img_path, label in raw_dataset.samples:
        image = Image.open(img_path).convert('RGB')  # 确保为 RGB

        # 原图（不增强）
        image_tensor = static_transform(image)
        all_images.append(image_tensor)
        all_labels.append(label)

        # 增强图像（增强多次）
        for _ in range(augment_times):
            aug_tensor = augment_transform(image)
            all_images.append(aug_tensor)
            all_labels.append(label)

    # 转成 TensorDataset
    all_images_tensor = torch.stack(all_images)
    all_labels_tensor = torch.tensor(all_labels)
    full_dataset = TensorDataset(all_images_tensor, all_labels_tensor)

    # 划分训练和测试集
    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    print('train_size:', train_size)
    print('test_size:', test_size)

    # DataLoader
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    # 定义 CNN 网络结构
    class OptimizedCNN(nn.Module):
        def __init__(self, num_classes=5):
            super(OptimizedCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.MaxPool2d(2)
            )

            self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256),
                nn.LeakyReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.Dropout(0.5),
                nn.Linear(128,num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = self.classifier(x)
            return x


    # 初始化模型
    model = OptimizedCNN().to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    # 学习率调度器：每 10 轮降低学习率
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    num_epochs = 500
    best_model_state_dict = None
    best_val_acc = 0.0
    patience = 50
    counter = 0
    # 训练过程
    print('=================开始训练=================')
    Loss_l, Train_acc_l, Val_acc_l = [], [], []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0

        # 训练集
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        scheduler.step()

        model.eval()
        correct = 0
        total = 0

        # 测试集
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        train_acc = correct_train / total_train
        Loss_l.append(total_loss)
        Train_acc_l.append(train_acc)
        Val_acc_l.append(val_acc)
        if epoch >= 100:
            if val_acc > best_val_acc and total_loss <= 1.5:
                best_val_acc = val_acc
                best_model_state_dict = model.state_dict()
                torch.save(best_model_state_dict, 'best_model.pt')
                print('best model saved')
                counter = 0  # 重置耐心计数器
            else:
                counter += 1

            if counter >= patience:
                print("Early stopping triggered.")
                break
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    matplotlib.use('TkAgg')

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']

    import matplotlib.pyplot as plt

    epochs = range(1, len(Loss_l) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, Loss_l, 'r-', label='Loss')
    plt.xlabel("Epoch", fontsize=13)
    plt.ylabel("Loss", fontsize=13)
    plt.title("训练损失曲线", fontsize=15)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, Train_acc_l, 'b-', label='Train Accuracy')
    plt.xlabel("Epoch", fontsize=13)
    plt.ylabel("Accuracy", fontsize=13)
    plt.title("训练准确率曲线", fontsize=15)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, Val_acc_l, 'g-', label='Validation Accuracy')
    plt.xlabel("Epoch", fontsize=13)
    plt.ylabel("Accuracy", fontsize=13)
    plt.title("验证准确率曲线", fontsize=15)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    print('模型已保存')
