import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import os

# 判断是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型定义
class Model(nn.Module):
    def __init__(self, num_classes=5):
        super(Model, self).__init__()
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
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

# 数据预处理 transform（验证 / 测试）
static_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

# 加载模型
model = Model()
model.load_state_dict(torch.load('./best_model.pt', map_location=device))
model.to(device)
model.eval()

# 加载数据并分类
data_path = './data'
raw_dataset = datasets.ImageFolder(root=data_path)
class_names = raw_dataset.classes  # 获取类别名
print(f"检测到的类别：{class_names}")

# 分类每张图片并输出路径+预测类别
print("\n开始分类：")
p = 0
with torch.no_grad():
    for img_path, _ in raw_dataset.samples:
        image = Image.open(img_path).convert('RGB')
        image_tensor = static_transform(image).unsqueeze(0).to(device)  # 添加 batch 维度

        output = model(image_tensor)
        pred_label = output.argmax(dim=1).item()
        pred_class = class_names[pred_label]

        print(f"{img_path} => 预测类别: {pred_class}")
        if pred_class in img_path:
            p += 1
print('模型在数据集上得分为:{:.2f}%'.format(p / len(raw_dataset) * 100))

