import cv2
import torch
import numpy as np
import mss
import yaml
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from threading import Thread
import pyautogui
import sys
import os
import random
import time
from pynput import keyboard
import datetime
import shutil
import IPython
import pandas
import psutil
import tqdm
import matplotlib
import seaborn

# 加载模型
print("加载模型中...")
model = torch.hub.load('.', 'custom', path='best.pt', source='local')
print("模型加载完成")

# 读取类别名称
print("读取类别名称中...")
with open('data/onmyoji.yaml', 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)
CLASS_NAMES = data['names']

# 读取中文名称映射
with open('data/onmyoji_name.yaml', 'r', encoding='utf-8') as f:
    name_data = yaml.safe_load(f)
Class_Name_To_Chinese = name_data['Class_Name_To_Chinese']

class TransparentWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.attributes("-topmost", True)
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-alpha", 0.3)  # 设置透明度
        self.root.overrideredirect(True)
        
        # 创建Canvas，使用系统透明色
        self.canvas = tk.Canvas(self.root, highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        
        # 设置透明背景
        transparent_color = '#000001'  # 使用一个几乎看不见的颜色
        self.canvas.configure(bg=transparent_color)
        self.root.wm_attributes("-transparentcolor", transparent_color)
        
        self.running = True
        
    def draw_detection(self, x1, y1, x2, y2, label, conf):
        # 绘制矩形框
        self.canvas.create_rectangle(x1, y1, x2, y2, outline='#ff69b4', width=2)
        # 绘制标签背景
        text = f'{Class_Name_To_Chinese.get(label, label)} {conf:.2f}'
        self.canvas.create_rectangle(x1, y1-25, x1+len(text)*10, y1, 
                                   fill='#ff69b4', outline='#ff69b4')
        # 绘制标签文字
        self.canvas.create_text(x1+5, y1-20, text=text, 
                              fill='black', anchor='w', font=('SimHei', 12))
        
    def clear_canvas(self):
        self.canvas.delete('all')
        
    def update(self):
        self.root.update()
        
def main():
    window = TransparentWindow()
    sct = mss.MSS()
    monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
    
    try:
        while window.running:
            # 捕获屏幕
            screen = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2RGB)
            
            # 运行检测
            results = model(frame)
            
            # 清除上一帧的绘制内容
            window.clear_canvas()
            
            # 绘制新的检测结果
            for detection in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
                if conf > 0.8:  # 置信度阈值
                    class_name = CLASS_NAMES[int(cls)]
                    window.draw_detection(
                        int(x1), int(y1), int(x2), int(y2),
                        class_name, conf
                    )
            
            # 更新窗口
            window.update()
            
            # 检查退出条件
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        window.root.destroy()

if __name__ == "__main__":
    main()