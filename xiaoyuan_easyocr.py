import easyocr
import numpy as np
import pyautogui
import time
import cv2


reader = easyocr.Reader(['en'], gpu = True)
def dayu():
    # 鼠标移动到起始位置
    pyautogui.moveTo(2008, 810)
    # 按下鼠标左键开始绘制
    pyautogui.mouseDown()
    # 画大于号
    pyautogui.moveTo(2301, 921)  # 第一部分
    pyautogui.moveTo(2049, 1075)  # 第二部分
    # 释放鼠标左键完成绘制
    pyautogui.mouseUp()
def xiaoyu():
    # 鼠标移动到起始位置
    pyautogui.moveTo(2348, 820)
    # 按下鼠标左键开始绘制
    pyautogui.mouseDown()
    pyautogui.moveTo(2053, 903)  # 第一部分
    pyautogui.moveTo(2353, 1036)  # 第二部分
    # 释放鼠标左键完成绘制
    pyautogui.mouseUp()

def denosied(path):
    # 加载图片
    image = cv2.imread(path)
    # 转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 将图像转换为浮动点形式，方便进行增强
    image_float = image.astype(np.float32)
    # 定义一个 alpha 值，调节对比度，值越大对比度增强越明显
    alpha = 2.0
    # 增强对比度
    enhanced_image = cv2.convertScaleAbs(image_float * alpha)
    # 保存预处理后的图片
    cv2.imwrite(path, gray_image)


def start():
    try:
        for i in range(999):
            path = './iii.png'
            im = pyautogui.screenshot(region=(2030, 300, 300, 100))
            im.save(path)
            denosied(path)
            result = reader.readtext(path, detail=0)
            print(result,end='')
            L = int(result[0])
            R = int(result[-1])
            if L > R:
                dayu()
                print('大于')
            elif L < R:
                xiaoyu()
                print('小于')
            else:
                dayu()
                time.sleep(0.30)
                xiaoyu()
            time.sleep(0.26)
    except Exception:
        pass
index = 1
while index:
    path_ = './ready.png'
    print('等待题目')
    img = pyautogui.screenshot(region=(2102, 640, 190, 85))
    img.save(path_)
    denosied(path_)
    result = reader.readtext(path_, detail=0)
    if len(result) != 0:
        if result[0] == 'Ready':
            print('开始')
            time.sleep(1.86)
            start()
            print('===================='
                  'End'
                  '====================')
            index = 0