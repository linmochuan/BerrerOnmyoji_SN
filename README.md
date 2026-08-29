<div align="center">

  <h1>lin RPA</h1>

  <p>
    <strong>基于 Python 的 Windows 桌面自动化与计算机视觉工具</strong>
  </p>

  <p>
    中文 | <a href="README_JP.md">日本語</a>
  </p>

</div>

---

# lin RPA

`lin RPA` 是一个基于 Python 开发的 Windows 桌面自动化工具。

项目最初用于解决重复性的桌面操作问题，在开发过程中逐步加入了**屏幕图像识别、YOLOv5 目标检测、PyAutoGUI 鼠标键盘控制、Excel 任务编排、后台线程执行和安全停止机制**等功能。

目前项目的主要执行方式是：

> **Excel 定义操作流程 → Python 读取任务 → 屏幕识别 → 执行鼠标/键盘操作 → 循环执行**

项目同时保留了早期开发过程中完成的 YOLOv5 实时目标检测与视觉自动化实验代码。

## 项目特点

* 使用**自制数据集训练 YOLOv5 目标检测模型**
* Python 实现实时屏幕目标检测
* 使用 PyAutoGUI 实现鼠标、键盘和截图操作
* 支持基于截图模板的屏幕识别
* 使用 Excel 配置桌面自动化流程，无需修改 Python 代码即可调整任务顺序
* 支持普通图片查找和持续性的全局图片查找
* 使用后台线程执行自动化任务，避免 Tkinter 主界面卡死
* 使用 `threading.Event` 实现任务停止控制
* 使用 JSON / YAML 管理程序配置、类别和动作
* 支持 PyInstaller 打包为 Windows 应用程序

---

# 一、项目结构

```text
lin RPA
│
├── lin_RPA.py              # RPA 主程序 / Tkinter GUI
├── rpa_readers.py          # 配置文件及 Excel 读取
├── rpa_operations.py       # 鼠标、截图、图片查找等桌面操作
├── rpa_executor.py         # Excel 任务执行及任务调度
│
├── J1.py                   # YOLO 屏幕检测实验
├── J1-1.py                 # YOLO 屏幕检测实验
├── W2.py                   # YOLO 实时检测实验
├── W3.py                   # YOLO 实时检测实验
├── jc.py                   # YOLO 检测结果可视化
│
├── pro.py                  # 早期 Excel 自动化程序
├── 屏幕.py                 # Windows 窗口及鼠标信息工具
├── 显示文字.py             # 屏幕文字覆盖工具
│
├── data/
│   ├── images/             # 模板图片
│   ├── onmyoji.yaml        # YOLO 类别配置
│   ├── onmyoji_name.yaml   # 类别名称映射
│   └── 3leader.yaml        # 类别与动作配置
│
├── models/                 # YOLOv5 模型相关代码
├── utils/                  # YOLOv5 工具代码
├── classify/               # YOLOv5 分类相关代码
├── segment/                # YOLOv5 分割相关代码
│
├── best.pt                 # 自训练 YOLOv5 模型
├── settings.json           # RPA 程序配置
├── requirements.txt
└── lin_RPA.spec
```

---

# 二、整体工作流程

当前 RPA 主程序从 `lin_RPA.py` 的 `main()` 开始。

```text
启动 lin_RPA.py
      │
      ├── 读取 settings.json / YAML 配置
      │
      ├── 创建 Tkinter GUI
      │
      └── 等待用户启动任务
                │
                ▼
          Excel 线性任务
                │
                ├── 读取 Excel
                │
                ├── 查找模板图片
                │
                ├── 判断是否找到目标
                │
                ├── 解析操作指令
                │
                ├── PyAutoGUI 执行操作
                │
                └── 进入下一任务
```

任务运行在后台线程中，Tkinter 主线程负责：

* 界面显示
* 用户操作
* 日志更新
* 开始 / 停止控制

因此，在等待图片或执行任务时，主界面不会因为后台任务阻塞而失去响应。

---

# 三、主要功能

## 1. Excel 驱动的自动化任务

项目将自动化流程放在 Excel 中配置，而不是直接写死在 Python 代码中。

Excel 每一行表示一个任务。

例如：

| 图片路径                     | 操作指令             |   查找时间 | 超时动作   |
| ------------------------ | ---------------- | -----: | ------ |
| `data/images/start.png`  | `等待=0.5，左键=1`    |     10 | `skip` |
| `data/images/button.png` | `偏移=10/5，二级左键=2` |      8 | `2`    |
| `data/images/notice.png` | `左键=1`           | `全局查找` |        |

程序按照 Excel 行顺序读取任务，并根据图片是否出现在屏幕上决定是否执行对应操作。

这种设计使自动化流程与程序代码分离：

```text
Excel
  │
  │ 定义任务
  ▼
Python
  │
  │ 读取 / 解析
  ▼
任务执行器
  │
  ├── 图片识别
  ├── 操作判断
  └── PyAutoGUI
```

修改操作流程时，只需要修改 Excel 配置，无需修改 Python 程序。

---

# 四、Excel 配置

## Excel 格式

程序从第 2 行开始读取任务。

| 列 | 名称   | 说明             |
| - | ---- | -------------- |
| A | 图片路径 | 要查找的模板图片       |
| B | 操作指令 | 找到图片后执行的操作     |
| C | 查找时间 | 图片查找时间，单位为秒    |
| D | 超时动作 | `skip` 或跳转到指定行 |

### 普通任务

例如：

```text
图片路径：data/images/start.png
操作指令：等待=0.5，左键=1
查找时间：10
超时动作：skip
```

程序会：

```text
查找 start.png
      ↓
找到？
  ├── 是 → 等待0.5秒 → 左键点击 → 下一行
  │
  └── 否 → 等待超时 → skip
```

## 全局查找

C 列填写：

```text
全局查找
```

后，该图片会在整个任务运行期间持续进行查找。

多个全局任务可以同时生效。

同时，为避免同一图片持续显示导致重复执行，程序会记录目标状态：

```text
图片出现
   ↓
执行一次
   ↓
图片仍然存在
   ↓
不重复执行
   ↓
图片消失
   ↓
再次出现
   ↓
重新执行
```

---

# 五、支持的操作指令

| 指令   | 示例                   | 功能                |
| ---- | -------------------- | ----------------- |
| 等待   | `等待=1.5`             | 等待指定时间            |
| 偏移   | `偏移=10/5`            | 后续坐标向右 10、向下 5 像素 |
| 左键单击 | `左键=2`               | 在目标中心点击 2 次       |
| 随机点击 | `二级左键=3`             | 在目标区域内随机点击 3 次    |
| 偏移点击 | `三级左键=10/20/1`       | 在指定偏移位置点击         |
| 左键按下 | `左键按下`               | 按下鼠标左键            |
| 左键释放 | `左键释放`               | 释放鼠标左键            |
| 右键按下 | `右键按下`               | 按下鼠标右键            |
| 右键释放 | `右键释放`               | 释放鼠标右键            |
| 全屏截图 | `屏幕截图`               | 保存当前屏幕            |
| 区域截图 | `区域屏幕截图=0/0/800/600` | 保存指定区域            |

多个操作可以使用中文逗号连接：

```text
偏移=15/8，等待=0.5，左键按下，等待=1，左键释放
```

程序会按照从左到右的顺序依次执行。

---

# 六、桌面图像识别

RPA 主流程使用 PyAutoGUI 的屏幕模板匹配功能：

```python
pyautogui.locateOnScreen()
```

基本流程：

```text
屏幕
 ↓
查找模板图片
 ↓
获得匹配区域
 ↓
计算中心坐标
 ↓
执行操作
```

支持：

* 普通图片查找
* 持续全局查找
* 多个全局目标
* 查找超时
* 匹配置信度设置
* 相对路径 / 绝对路径
* 截图保存

---

# 七、YOLOv5 目标检测

除了基于模板图片的 RPA，本项目还包含基于 YOLOv5 的目标检测实验。

## 模型训练

项目使用**自行制作的数据集训练 YOLOv5 模型**，最终得到：

```text
best.pt
```

模型用于识别特定界面中的目标。

训练过程涉及：

```text
数据收集
   ↓
数据标注
   ↓
训练集 / 验证集划分
   ↓
YOLOv5 模型训练
   ↓
模型验证
   ↓
best.pt
   ↓
Python 推理
```

模型并非直接使用第三方训练好的目标检测权重，而是根据项目实际需求制作数据集并进行训练。

> 注：本项目的 YOLOv5 代码基础来自 YOLOv5 开源项目，本人主要完成数据集、训练配置、模型使用以及上层应用逻辑。

---

# 八、YOLO 实时屏幕检测

项目早期实验程序可以使用：

* `mss`
* `torch`
* `OpenCV`
* `PyYAML`
* `PyAutoGUI`
* `Tkinter`

实时读取屏幕并进行目标检测。

基本流程：

```text
PC屏幕
   ↓
mss 截图
   ↓
YOLOv5 推理
   ↓
获取检测结果
   ↓
类别 / 置信度 / 坐标
   ↓
透明窗口绘制检测框
```

检测结果可以在屏幕上实时显示：

```text
┌──────────────────────────┐
│                          │
│      ┌────────────┐      │
│      │   Target   │      │
│      │  0.93      │      │
│      └────────────┘      │
│                          │
└──────────────────────────┘
```

---

# 九、基于视觉识别的自动化

YOLO 检测结果也曾用于驱动自动化操作。

基本流程：

```text
屏幕截图
   ↓
YOLOv5
   ↓
目标检测
   ↓
类别判断
   ↓
目标坐标计算
   ↓
动作决策
   ↓
PyAutoGUI
   ↓
鼠标 / 键盘操作
```

部分操作会根据目标区域生成随机坐标，而不是始终点击固定中心点。

例如：

```text
目标区域
┌─────────────────┐
│       ·         │
│    ·  ·  ·      │
│  ·  ·  ·  ·     │
│    ·  ●  ·      │
│       ·         │
└─────────────────┘
```

通过随机采样选择目标区域中的操作位置，用于实现更加灵活的 GUI 自动化操作。

---

# 十、RPA 模块设计

当前 RPA 程序主要由以下模块组成：

```text
lin_RPA.py
    │
    ├── rpa_readers.py
    │       └── 配置 / Excel / YAML
    │
    ├── rpa_operations.py
    │       └── 鼠标 / 截图 / 图片查找
    │
    └── rpa_executor.py
            └── 任务执行 / 循环 / 停止控制
```

## 1. `lin_RPA.py`

负责：

* Tkinter 图形界面
* Excel 文件选择
* 工作表选择
* 置信度设置
* 运行时间设置
* 截图目录设置
* 开始 / 停止
* 日志显示

---

## 2. `rpa_readers.py`

负责读取和保存配置。

主要使用：

* `json`
* `pathlib`
* `openpyxl`
* `PyYAML`

功能包括：

* 读取 `settings.json`
* 读取 YAML
* 读取 Excel
* 解析图片路径
* 保存用户配置
* 校验置信度

置信度限制在：

```text
0.1 ～ 1.0
```

如果配置异常，则使用默认值：

```text
0.8
```

---

## 3. `rpa_operations.py`

负责封装桌面操作。

主要使用：

* `PyAutoGUI`
* `pynput`
* `numpy`
* `threading`

包括：

* 鼠标点击
* 鼠标按下 / 释放
* 截图
* 图片查找
* 随机点击
* 全局快捷键
* 停止信号

---

## 4. `rpa_executor.py`

负责组织自动化任务。

主要功能：

* Excel 任务读取
* 图片查找
* 全局图片监听
* 操作指令解析
* 任务循环
* 超时处理
* 后台线程执行
* 停止控制

---

# 十一、异步执行与安全停止

为了避免 GUI 因自动化任务阻塞，Excel 任务使用后台线程执行。

结构：

```text
Tkinter 主线程
│
├── GUI
├── 按钮
├── 日志
│
└── start_async()
       │
       ▼
  ExcelExecutor
       │
       └── 后台线程
```

程序使用：

```python
threading.Thread
threading.Event
```

实现任务控制。

## StopController

停止信号由 `threading.Event` 管理。

以下操作共享同一个停止机制：

* GUI 停止按钮
* 窗口关闭
* 全局快捷键

收到停止信号后：

```text
StopController
      ↓
threading.Event.set()
      ↓
任务循环检查停止状态
      ↓
后台线程自然退出
```

程序不会强制杀死执行中的进程。

---

# 十二、日志与错误处理

程序会记录自动化任务运行状态，包括：

* 图片查找成功
* 图片查找失败
* 查找超时
* 操作执行
* 配置错误
* 任务停止

为了避免日志刷屏：

* 全局查找不会持续输出轮询日志
* 连续重复的日志消息会进行合并

启动时也会检查：

* Excel 文件
* 工作表
* 图片路径
* 置信度
* 运行时间
* 配置文件

---

# 十三、配置文件

## `settings.json`

示例：

```json
{
    "confidence": 0.8,
    "excel_path": "D:/rpa/tasks.xlsx",
    "sheet_name": "Sheet1",
    "screenshot_path": "D:/rpa/screenshots",
    "hotkey": ["CTRL", "Q"],
    "model_path": "best.pt"
}
```

## YAML

主要包括：

```text
data/onmyoji.yaml
data/onmyoji_name.yaml
data/3leader.yaml
```

分别用于：

* YOLO 类别定义
* 类别名称映射
* 识别类别与动作映射

---

# 十四、主要技术栈

| 技术          | 用途           |
| ----------- | ------------ |
| Python      | 主要开发语言       |
| YOLOv5      | 目标检测模型       |
| PyTorch     | 模型训练与推理      |
| OpenCV      | 图像处理         |
| MSS         | 屏幕截图         |
| PyAutoGUI   | 鼠标、键盘和屏幕操作   |
| pynput      | 全局快捷键及输入监听   |
| Tkinter     | Windows GUI  |
| OpenPyXL    | Excel 文件处理   |
| PyYAML      | YAML 配置      |
| NumPy       | 数值计算与随机采样    |
| Threading   | 后台任务及停止控制    |
| PyInstaller | Windows 程序打包 |

---

# 十五、安装

建议使用 Python 虚拟环境。

```bash
python -m venv .venv
```

Windows：

```bash
.venv\Scripts\activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

根据实际环境补充安装：

```bash
pip install torch opencv-python mss pyautogui pynput openpyxl PyYAML numpy
```

---

# 十六、运行

启动 RPA：

```bash
python lin_RPA.py
```

启动后：

1. 选择 Excel 文件
2. 选择工作表
3. 设置图片目录
4. 设置检测置信度
5. 设置运行时间
6. 点击开始
7. 程序在后台执行 Excel 任务

---

# 十七、Windows 打包

项目提供：

```text
lin_RPA.spec
```

可以使用 PyInstaller 打包：

```bash
pyinstaller lin_RPA.spec
```

生成：

```text
dist/
└── lin_RPA/
```

打包后的程序需要保留：

```text
best.pt
settings.json
data/
Python 模块
```

等相关资源。

建议复制整个 `dist/lin_RPA/` 目录，而不是只复制 exe 文件。

---

# 十八、项目开发经历

这个项目经历了从简单自动化脚本到模块化 RPA 工具的逐步演进。

```text
简单 GUI 自动化
      ↓
PyAutoGUI 图片识别
      ↓
Excel 驱动任务
      ↓
任务执行器
      ↓
后台线程
      ↓
安全停止机制
      ↓
YOLOv5 自定义模型训练
      ↓
实时屏幕目标检测
      ↓
视觉识别驱动自动化
```

通过这个项目，主要实践了：

* Python 桌面应用开发
* GUI 开发
* 文件及配置管理
* Excel 数据处理
* 图像识别
* YOLO 模型训练与推理
* 自动化操作
* 多线程
* 事件驱动的任务控制
* 模块化设计
* Windows 程序打包

---

# 十九、后续计划

目前正在学习 **FastAPI**，计划将现有的 YOLOv5 推理能力进一步封装成 Web API。

目标结构：

```text
Client
   │
   │ POST /predict
   ▼
FastAPI
   │
   ▼
YOLOv5
   │
   ▼
目标检测
   │
   ▼
JSON Response
```

后续计划包括：

* [ ] 使用 FastAPI 封装 YOLO 推理 API
* [ ] 增加图片上传接口
* [ ] 增加识别历史记录
* [ ] 使用 PostgreSQL / MySQL 保存数据
* [ ] 使用 Docker 部署
* [ ] 增加 pytest 自动化测试
* [ ] 增加 Web 前端用于展示检测结果

---

# 二十、说明

本项目最初用于个人桌面自动化和计算机视觉学习。

其中 YOLOv5 部分基于开源 YOLOv5 项目，本仓库主要展示个人完成的数据集制作、模型训练、模型使用以及上层 Python 自动化程序。

项目中的部分旧版脚本属于开发过程中的实验代码，与当前 `lin_RPA.py` 主程序并非全部保持一致。

目前推荐使用：

```bash
python lin_RPA.py
```

运行当前版本的 RPA 主程序。
