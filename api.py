import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
'''
将模型功能封装成api一次可以识别一张图片中花的种类
'''
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

# 主分类函数
def flowers_api(image_path, model_path='./best_model.pt', data_path='./data', image_size=(128, 128)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 图像预处理 pipeline
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    try:
        raw_dataset = datasets.ImageFolder(root=data_path)
        class_names = raw_dataset.classes
    except Exception as e:
        print(f"[错误] 读取类别失败: {e}")
        return None

    try:
        # 加载模型
        model = Model(num_classes=len(class_names))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"[错误] 加载模型失败: {e}")
        return None

    try:
        # 加载图片并预测
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            pred_label = output.argmax(dim=1).item()
            pred_class = class_names[pred_label]
        return pred_class
    except Exception as e:
        print(f"[错误] 图片分类失败: {e}")
        return None

# 示例调用
# result = flowers_api('./data/daisy/2482982436_a2145359e0_n.jpg')
# print("预测结果:", result)
