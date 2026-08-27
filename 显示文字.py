import tkinter as tk

# 创建主窗口
root = tk.Tk()
root.attributes("-fullscreen", True)  # 设置窗口全屏
root.attributes("-topmost", True)  # 将窗口置于顶层
root.overrideredirect(True)  # 移除窗口装饰

# 设置一个不常见的背景颜色
transparent_color = '#00FF00'
root.config(bg=transparent_color)

# 将这个背景颜色设置为透明
root.wm_attributes("-transparentcolor", transparent_color)

# 创建一个标签显示信息
message = "This is a transparent overlay"
label = tk.Label(root, text=message, font=("Helvetica", 12), fg="black", bg=transparent_color)

# 将标签放置在左下角
label.place(x=10, y=root.winfo_screenheight() - 100)  # 根据需要调整y坐标

# 主循环
root.mainloop()