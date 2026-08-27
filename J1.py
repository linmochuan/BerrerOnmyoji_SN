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
import win32gui
import win32con
import math

print("程序开始")
# 检查环境和路径
log_list = []  # 初始化 log_list



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
    global log_list, root
    log_text = "\n".join(log_list)
    # 更新标签文本
    shadow.config(text=log_text)
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
with open('data\onmyoji.yaml', 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)
CLASS_NAMES = data['names']
print("类别名称读取完成。可以正常使用")
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
action_clicked = False
action_x1, action_y1, action_x2, action_y2 = 0, 0, 0, 0
special_labels = ['Mobs', 'Boss']

# 添加全局变量
s_x, s_y = None, None  # 用于存储Action点击位置

def g_r(center, low, high, size=1): 
    std_dev = (high - low) / 6.0
    values = np.random.normal(loc=center, scale=std_dev, size=size)
    values = np.clip(values, low, high)
    return values

# 点击规律-正态分布1
def click (class_name, Shiji_x, Shiji_y):
    pyautogui.moveTo(Shiji_x, Shiji_y)
    pyautogui.mouseDown()
    time.sleep(random.uniform(0.06464266777038574, 0.1))# 点击时长
    pyautogui.mouseUp()
    chinese_name = Class_Name_To_Chinese.get(class_name, class_name)
    now = datetime.datetime.now()
    add_log(f'时间:{now.strftime("%H:%M:%S")}单击：【{chinese_name}】')

#随机点击时长
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
    global running, action_clicked
    window = TransparentWindow()
    with mss.MSS() as sct:
        while running:
            # 截取屏幕
            img = np.array(sct.grab(monitor))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 用YOLOv5模型进行检测
            results = model(img_rgb)
            
            # 处理检测结果
            for detection in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
                if conf > conf_threshold:
                    class_name = CLASS_NAMES[int(cls)]
                    window.draw_detection(
                        int(x1), int(y1), int(x2), int(y2),
                        class_name, conf
                    )
            
            # 更新窗口
            window.root.update()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

class TransparentWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.attributes("-topmost", True)
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-alpha", 0.3)
        self.root.overrideredirect(True)
        
        # 创建Canvas
        self.canvas = tk.Canvas(self.root, highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        
        # 设置透明背景
        transparent_color = '#000001'
        self.canvas.configure(bg=transparent_color)
        self.root.wm_attributes("-transparentcolor", transparent_color)
        
        self.target_window = None
        self.running = True
        # 添加一个变量来跟踪所有绘制的图形ID
        self.current_drawings = []

    def get_window_info(self, x, y):
        """获取指定坐标所在窗口的信息"""
        hwnd = win32gui.WindowFromPoint((x, y))
        if hwnd:
            window_text = win32gui.GetWindowText(hwnd)
            rect = win32gui.GetWindowRect(hwnd)
            return {
                'hwnd': hwnd,
                'title': window_text,
                'rect': rect
            }
        return None

    def is_point_in_window(self, x, y, window_info):
        """检查点是否在指定窗口内"""
        if window_info and 'rect' in window_info:
            left, top, right, bottom = window_info['rect']
            return left <= x <= right and top <= y <= bottom
        return False

    def draw_window_border(self, window_info):
        # 清除所有之前绘制的图形
        for item_id in self.current_drawings:
            self.canvas.delete(item_id)
        self.current_drawings.clear()
        
        if window_info:
            left, top, right, bottom = window_info['rect']
            
            # 获取当前时间用于颜色渐变
            t = time.time()
            r = int(255 * (math.sin(t) + 1) / 2)
            g = int(255 * (math.sin(t + 2.09) + 1) / 2)
            b = int(255 * (math.sin(t + 4.18) + 1) / 2)
            color = f'#{r:02x}{g:02x}{b:02x}'
            
            # 保存所有新绘制图形的ID
            rect_id = self.canvas.create_rectangle(
                left, top, right, bottom,
                outline=color,
                width=10
            )
            self.current_drawings.append(rect_id)
            
            bg_id = self.canvas.create_rectangle(
                left, top-30, left+300, top,
                fill=color,
                outline=color
            )
            self.current_drawings.append(bg_id)
            
            text_id = self.canvas.create_text(
                left+5, top-15,
                text=f"这是队长的窗口",
                fill='white',
                anchor='w',
                font=('SimHei', 12, 'bold')
            )
            self.current_drawings.append(text_id)

    def draw_detection(self, x1, y1, x2, y2, label, conf):
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # 如果检测到Action标签
        if label == 'Action':
            window_info = self.get_window_info(center_x, center_y)
            if window_info:
                self.target_window = window_info
                self.canvas.delete('window_border')  # 清除之前的边框
                self.draw_window_border(window_info)  # 绘制新的边框
                now = datetime.datetime.now()
                add_log(f'{now.strftime("%H:%M:%S")}:记录窗口: {window_info["title"]}')

        # 对于Mobs和Boss的处理
        elif label in ['Mobs', 'Boss']:
            if self.target_window and self.is_point_in_window(center_x, center_y, self.target_window):
                Shiji_x = g_r(center_x, x1, x2, 1)
                Shiji_y = g_r(center_y, y1, y2, 1)
                click(label, Shiji_x, Shiji_y)
                time.sleep(0.5)

# 创建一个透明的全屏窗口用于显示日志
root = tk.Tk()
root.attributes("-fullscreen", True)  # 设置窗口全屏
root.attributes("-topmost", True)  # 将窗口置于顶层
root.overrideredirect(True)  # 移除窗口装饰

# 透明背景
transparent_color = "#ffffff"
root.config(bg=transparent_color)
root.wm_attributes("-transparentcolor", transparent_color)

# 标签
shadow = tk.Label(root, text="", font=("微软雅黑", 15, "bold"), fg="#000000", bg=transparent_color)
shadow.place(x=11, y=root.winfo_screenheight() - 349)  # 阴影
label = tk.Label(root, text="", font=("微软雅黑", 15, "bold"), fg="#000000", bg=transparent_color)
label.place(x=10, y=root.winfo_screenheight() - 350)  # 正文
now = datetime.datetime.now()
add_log(f'{now.strftime("%H:%M:%S")}:程序开始')
add_log('请多多支持荇子~')
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
