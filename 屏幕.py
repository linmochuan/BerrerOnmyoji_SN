import tkinter as tk
import win32gui
import win32api
import time
from threading import Thread
import math

class WindowDetector:
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
        
        self.running = True
        
        # 启动鼠标跟踪线程
        self.track_thread = Thread(target=self.track_mouse, daemon=True)
        self.track_thread.start()

    def get_window_info(self, x, y):
        """获取指定坐标所在窗口的信息"""
        hwnd = win32gui.WindowFromPoint((x, y))
        if hwnd:
            window_text = win32gui.GetWindowText(hwnd)
            class_name = win32gui.GetClassName(hwnd)
            rect = win32gui.GetWindowRect(hwnd)
            return {
                'hwnd': hwnd,
                'title': window_text,
                'class': class_name,
                'rect': rect
            }
        return None

    def draw_window_info(self, window_info):
        """绘制窗口信息和边框"""
        self.canvas.delete('all')  # 清除之前的绘制
        
        if window_info:
            # 获取窗口坐标
            left, top, right, bottom = window_info['rect']
            
            # 获取当前时间用于颜色渐变
            t = time.time()
            # 使用正弦函数创建RGB颜色渐变效果
            r = int(255 * (math.sin(t) + 1) / 2)  # 0-255 红色
            g = int(255 * (math.sin(t + 2.09) + 1) / 2)  # 0-255 绿色，相位差 2.09 (约120度)
            b = int(255 * (math.sin(t + 4.18) + 1) / 2)  # 0-255 蓝色，相位差 4.18 (约240度)
            color = f'#{r:02x}{g:02x}{b:02x}'
            
            # 绘制窗口边框
            self.canvas.create_rectangle(
                left, top, right, bottom,
                outline=color,
                width=10  # 边框宽度从5增加到10
            )
            
            # 准备显示文本
            text = f"窗口: {window_info['title']}\n"            
            text += f"坐标: ({left}, {top}, {right}, {bottom})"
            
            # 绘制文本背景
            text_bg_height = 60
            self.canvas.create_rectangle(
                left, top-text_bg_height, left+300, top,
                fill=color,
                outline=color
            )
            
            # 绘制文本
            y_offset = 20
            for line in text.split('\n'):
                self.canvas.create_text(
                    left+5, top-text_bg_height+y_offset,
                    text=line,
                    fill='white',
                    anchor='w',
                    font=('SimHei', 12, 'bold')
                )
                y_offset += 20

    def track_mouse(self):
        """跟踪鼠标位置并更新窗口信息"""
        while self.running:
            try:
                # 获取鼠标位置
                x, y = win32api.GetCursorPos()
                # 获取窗口信息
                window_info = self.get_window_info(x, y)
                # 绘制信息
                self.draw_window_info(window_info)
                # 更新窗口
                self.root.update()
                # 短暂休眠以减少CPU使用
                time.sleep(0.1)
            except Exception as e:
                print(f"Error: {e}")
                continue

    def run(self):
        try:
            self.root.mainloop()
        finally:
            self.running = False
            self.root.destroy()

def main():
    detector = WindowDetector()
    detector.run()

if __name__ == "__main__":
    main()
