import cv2
import torch
import mss
import pyautogui
import sys
import os
import yaml
import random
import time
from pynput import keyboard
import tkinter as tk
from threading import Thread
import datetime
import numpy as np
import shutil
import IPython
import pandas
import psutil
import tqdm
import matplotlib
import seaborn

print("程序开始")
# 检查环境和路径
log_list = []  # 初始化 log_list

# 设置过期时间
expiration_date = datetime.datetime(2024, 10, 28)  # 将这里的日期改为您希望程序停止运行的日期
# 检查当前日期
current_date = datetime.datetime.now()
if current_date > expiration_date:
    print("程序已过期,无法继续使用。")
    input("按任意键退出...")
    sys.exit(1)

if getattr(sys, 'frozen', False):
    m_a = sys._MEIPASS
else:
    m_a = os.getcwd()

# 检查模型文件
model_path = os.path.join(m_a, 'best.pt')
if not os.path.exists(model_path):
    print(f"模型文件不存在: {model_path}")
    input()
    sys.exit(1)

# 加载 YOLOv5 模型
print("加载模型中（加载速度和电脑配置有关）...")
model = torch.hub.load('.', 'custom', path=model_path, source='local', force_reload=True)
print("模型加载完成。")
# 输出目录
output_dir = os.path.join(os.path.dirname(__file__), 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 定义显示日志的函数
def display_log():
    global log_list
    log_text = "\n".join(log_list)
    label.config(text=log_text)

# 在透明窗口的左下角显示日志
def add_log(message):
    global log_list
    if len(log_list) >= 10:
        log_list.pop(0)
    log_list.append(message)
    display_log()

# 从 data.yaml 文件读取类别名称
print("读取类别名称中...")
with open('data/onmyoji.yaml', 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)
CLASS_NAMES = data['names']
print("读取文件name...")
with open('data/onmyoji_name.yaml', 'r', encoding='utf-8') as f:
    name_data = yaml.safe_load(f)
Class_Name_To_Chinese = name_data['Class_Name_To_Chinese']
print("类别名称和中文对应关系读取完成。")
with open('data/3leader.yaml', 'r', encoding='utf-8') as f:
    actions_config = yaml.safe_load(f)['actions']

# 设置屏幕截图区域
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
# 设置检测阈值
conf_threshold = 0.5
# 操作日志列表
log_list = []

# 点击规律-正态分布
def g_r(center, low, high, size=1): 
    std_dev = (high - low) / 6.0
    values = np.random.normal(loc=center, scale=std_dev, size=size)
    values = np.clip(values, low, high)
    return values

#随机点击时长
def click (class_name, Shiji_x, Shiji_y):
    pyautogui.moveTo(Shiji_x, Shiji_y)
    pyautogui.mouseDown()
    time.sleep(random.uniform(0.06464266777038574, 0.1))# 点击时长
    pyautogui.mouseUp()
    chinese_name = Class_Name_To_Chinese.get(class_name, class_name)
    now = datetime.datetime.now()
    add_log(f'时间:{now.strftime("%H:%M:%S")}单击：【{chinese_name}】')

def gouxie (img_rgb, x1, x2, y1, y2, name):
    GouXie_results = model(img_rgb[int(y1):int(y2), int(x1):int(x2)])
    for g_d in GouXie_results.xyxy[0]:
        gx_1, gy_1, gx2, gy2, gconf, gcls = g_d.cpu().numpy()  # 将检测结果移至 CPU
        if gconf > conf_threshold and CLASS_NAMES[int(gcls)] == name:
            g_center_x, g_center_y = int(gx_1 + gx2) / 2 + x1, int(gy_1 + gy2) / 2 + y1
            click(name, g_center_x, g_center_y)

# 定义检测和显示函数
running = True
def detect_and_display():
    global running
    with mss.mss() as sct:
        while running:
            # 截取屏幕
            img = np.array(sct.grab(monitor))
            # 转换颜色空间，从BGR到RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 使用YOLOv5模型进行检测
            results = model(img_rgb)
            # 临时列表保存当前帧检测到的标签
            detected_labels = []
            # 处理检测结果
            for detection in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
                if conf > conf_threshold:
                    cls_index = int(cls)
                    if cls_index < len(CLASS_NAMES):
                        class_name = CLASS_NAMES[cls_index]
                        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        c_x, c_y = int(x2 - x1), int(y2 - y1)

                        if class_name in actions_config:
                            action = actions_config[class_name]
                            if action.get('special'):
                                if class_name == 'GouXie':
                                    gouxie(img_rgb, x1, x2, y1, y2, 'X_pink')
                            else:
                                x_params = action['x']
                                y_params = action['y']
                                Shiji_x = g_r(eval(x_params[0]), eval(x_params[1]), eval(x_params[2]), 1)
                                Shiji_y = g_r(eval(y_params[0]), eval(y_params[1]), eval(y_params[2]), 1)
                                click(class_name, Shiji_x, Shiji_y)

# 创建一个透明的全屏窗口用于显示日志
root = tk.Tk()
root.attributes("-fullscreen", True)  # 设置窗口全屏
root.attributes("-topmost", True)  # 将窗口置于顶层
root.overrideredirect(True)  # 移除窗口装饰

# 透明背景
transparent_color = '#00FF00'
root.config(bg=transparent_color)
root.wm_attributes("-transparentcolor", transparent_color)

# 标签
# shadow = tk.Label(root, text="", font=("微软雅黑", 15, "bold"), fg="#ffffff", bg=transparent_color)
# shadow.place(x=11, y=root.winfo_screenheight() - 349)  # 阴影
label = tk.Label(root, text="", font=("微软雅黑", 14, "bold"), fg="#311417", bg=transparent_color)
label.place(x=10, y=root.winfo_screenheight() - 350)  # 正文
now = datetime.datetime.now()
add_log(f'{now.strftime("%H:%M:%S")}:程序开始')
add_log('御魂模式')
add_log('请多多支持水年多吃饭~')
add_log('有问题邮箱联系：Linwateryear@outlook.com')
add_log('邮件主题为：快去修bug')
add_log('如果识别不到图片请多截几张图一并发送，图片要整个游戏界面')
# 主
thread = Thread(target=detect_and_display)
thread.start()

# 用于跟踪 Ctrl 键的状态
ctrl_pressed = False

def on_press(key):
    global running, ctrl_pressed
    try:
        if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
            ctrl_pressed = True  # 按下Ctrl键时设置为True
        elif key.char == 'q' and ctrl_pressed:  # 检查是否同时按下了Ctrl和Q
            print("Ctrl + Q 被按下，程序停止。")
            running = False
            root.quit()  # 退出Tkinter主循环
            return False  # 停止键盘监听器
    except AttributeError:
        pass

def on_release(key):
    global ctrl_pressed
    if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
        ctrl_pressed = False  # 松开Ctrl键时重置状态

# 键盘监听器
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# 主循环
root.mainloop()
