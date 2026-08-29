<div align="center">

  <a href="https://github.com/linmochuan/BerrerOnmyoji_SN" target="_blank">
    <h1>lin RPA</h1>
  </a>

  <p>
    中文 | <a href="README_JP.md">日本語</a>
  </p>

  <br>

  <div>Windows 桌面 Excel 自动化工具</div>

</div>

# lin RPA

`lin RPA` 是一个面向 Windows 桌面程序的 Python 自动化工具，将屏幕识别、图片模板查找、鼠标操作、Excel 任务编排和 YOLO 模型识别组合在一起。

当前主界面提供一条执行路线：

- **Excel 线性任务**：按照 Excel 行顺序查找指定截图，找到后执行该行配置的操作。

Excel 任务运行在后台线程中，主界面不会因为等待图片或执行操作而卡住。停止按钮、窗口关闭和配置的全局快捷键会向后台任务发送停止信号。

## 一、整体流程

程序从 `lin_RPA.py` 的 `main()` 开始：

```text
启动 lin_RPA.py
    |
    +-- 读取 settings.json、data/*.yaml
    |
    +-- 创建 Tkinter 界面并恢复保存的配置
    |
    +-- 等待用户启动 Excel 任务
          |
          +-- Excel 线性任务
          |     +-- 校验 Excel 和工作表
          |     +-- 后台读取任务行
          |     +-- 循环查找模板图片
          |     +-- 找到后按顺序执行操作
          |     +-- 读完后回到第 2 行继续
          |
```

### 1. 启动阶段

```bash
python lin_RPA.py
```

启动时的具体步骤：

1. `rpa_readers.project_root()` 确定资源目录。源码运行时使用当前项目目录，PyInstaller 运行时使用打包目录。
2. `read_app_config()` 读取 `settings.json` 和 `data/` 下的 YAML 配置。
3. 从 `settings.json` 读取检测置信度；配置缺失或格式错误时使用默认值 `0.8`。
4. 读取 `onmyoji.yaml` 的类别名称、`onmyoji_name.yaml` 的中文名称映射和 `3leader.yaml` 的动作配置。
5. 创建 Tkinter 界面，包括 Excel 路径、工作表、截图目录、置信度、运行时间和控制按钮。
6. 恢复上次保存的 Excel 路径、截图目录和快捷键。
7. 如果存在快捷键，则启动全局键盘监听器。
8. 进入 Tkinter `mainloop()`，等待用户操作。

模块导入不会自动加载模型、启动线程、启动键盘监听或操作鼠标；这些行为只会在明确启动后发生。

## 二、使用教程

点击“开始”后，流程如下：

1. 读取界面中的 Excel 路径和工作表名称。
2. 检查文件是否存在，并确认工作表有效。
3. 读取运行时间和置信度，并保存到 `settings.json`。
4. 创建 `StopController`，用于在线程之间传递停止信号。
5. 创建 `DesktopOperations`，统一处理图片查找、鼠标和截图。
6. 创建 `ExcelExecutor`，通过 `start_async()` 放入后台线程。
7. 从 Excel 第 2 行开始读取任务，每行读取前四列。
8. C 列填写 `全局查找` 的任务会在整个运行期间持续查找，所有全局任务同时生效。
9. 普通任务在查找时间内反复调用 `pyautogui.locateOnScreen()` 查找模板图片。
10. 找到图片后计算图片中心坐标，按中文逗号分割并依次执行操作。
11. 当前行完成后进入下一行，读完最后一行后重新从第 2 行开始。
12. 运行时间结束或收到停止信号后，后台线程退出。

### Excel 格式

第 1 行表头，程序从第 2 行开始读取：

| 列 | 名称 | 说明 |
| --- | --- | --- |
| A | 图片路径 | 要查找的模板图片路径 |
| B | 操作指令 | 多条指令使用中文逗号 `，` 分隔 |
| C | 查找时间 | 单位为秒，留空默认 10 秒；填写 `全局查找` 后持续全局查找 |
| D | 超时动作 | `skip` 跳过；数字表示跳转到指定行 |

示例：

| 图片路径 | 操作指令 | 查找时间 | 超时动作 |
| --- | --- | ---: | --- |
| `data/images/start.png` | `等待=0.5，左键=1` | 10 | `skip` |
| `data/images/button.png` | `偏移=10/5，二级左键=2` | 8 | `2` |
| `data/images/notice.png` | `左键=1` | `全局查找` | |

相对路径以项目根目录为基准，也可以填写绝对路径。

### 操作指令

| 指令 | 示例 | 作用 |
| --- | --- | --- |
| 等待 | `等待=1.5` | 暂停指定秒数 |
| 偏移 | `偏移=10/5` | 后续坐标向右 10、向下 5 像素 |
| 左键单击 | `左键=2` | 在图片中心单击 2 次 |
| 随机点击 | `二级左键=3` | 在图片区域内随机点击 3 次 |
| 偏移点击 | `三级左键=10/20/1` | 在中心偏移 `(10,20)` 的位置点击 1 次 |
| 左键按下 | `左键按下` | 按下左键 |
| 左键释放 | `左键释放` | 释放左键 |
| 右键按下 | `右键按下` | 按下右键 |
| 右键释放 | `右键释放` | 释放右键 |
| 全屏截图 | `屏幕截图` | 保存当前屏幕 |
| 区域截图 | `区域屏幕截图=0/0/800/600` | 保存指定区域 |

例如：

```text
偏移=15/8，等待=0.5，左键按下，等待=1，左键释放
```

表示先移动坐标，等待半秒，按下左键，再等待一秒，最后释放左键。

## 三、RPA 功能模块

RPA 部分将配置管理、Excel 流程编排、屏幕识别、桌面操作、异步执行和停止控制集成到一个 Windows 自动化工具中，可以把“看到指定画面后执行一组操作”的人工流程转换为可重复运行的 Excel 任务。

### 1. 图形界面模块

由 `lin_RPA.py` 和 `tkinter` 实现：

- 选择 Excel 文件和工作表，并刷新工作表列表。
- 设置图片匹配置信度、运行时长和截图保存目录。
- 提供开始、停止和窗口关闭操作。
- 在界面中显示带时间的运行日志，并安全更新后台线程消息。

### 2. 配置和任务读取模块

由 `rpa_readers.py` 实现，集成 `json`、`openpyxl`、`PyYAML` 和 `pathlib`：

- 读取和保存 `settings.json` 中的 Excel 路径、工作表、置信度、截图目录、快捷键和模型路径。
- 将置信度限制在 `0.1` 到 `1.0`，配置错误时使用默认值 `0.8`。
- 读取 Excel 第 2 行开始的 A 到 D 列任务。
- 支持图片相对路径、绝对路径、用户目录和环境变量展开。

### 3. Excel 流程编排模块

由 `rpa_executor.py` 的 `ExcelExecutor` 实现：

- 普通任务根据 A 列图片路径查找屏幕模板。
- 找到模板后，以匹配区域中心作为操作坐标。
- B 列支持使用中文逗号分隔的多条操作指令，并按顺序执行。
- D 列支持跳过当前任务或跳转到指定行。
- 任务读到末尾后从第 2 行重新开始，适合重复性桌面流程。

### 4. 图片识别和全局查找模块

由 `rpa_operations.py` 和 `rpa_executor.py` 集成 `PyAutoGUI` 完成：

- 使用 `pyautogui.locateOnScreen()` 查找模板图片。
- 普通任务只在 C 列指定的查找时间内查找。
- C 列填写 `全局查找` 后，图片会在程序运行期间持续查找。
- 可以配置多个全局查找目标，同时处理多个图片。
- 同一图片持续显示时只触发一次，消失后再次出现才重新执行，避免重复点击。

### 5. 桌面操作模块

由 `DesktopOperations` 和 `random_position()` 实现，集成 `PyAutoGUI`、`numpy` 和 `pynput`：

- 鼠标左键、右键按下和释放。
- 固定坐标点击、重复点击、偏移点击和区域内随机点击。
- 全屏截图和指定区域截图，并自动创建保存目录。
- 监听用户配置的停止快捷键。

### 6. 异步执行和安全停止模块

由 `start_async()` 和 `StopController` 实现，使用 `threading.Thread` 和 `threading.Event`：

- Excel 任务在后台线程运行，Tkinter 主界面保持响应。
- 停止按钮、窗口关闭和全局快捷键共享同一个停止事件。
- 普通查找、全局查找、重复点击和任务循环都会检查停止状态。
- 线程收到停止信号后自然退出，不强制结束进程。

### 7. 日志和错误处理模块

- 普通图片查找输出成功、失败和超时信息。
- 全局查找不输出持续轮询日志，避免日志刷屏。
- 连续重复的日志消息会合并显示。
- 置信度、运行时间和路径异常会在启动时提示。

### 8. RPA 集成后的实际效果

```text
用户在 GUI 设置参数
    -> 读取 settings.json 和 Excel
    -> 启动后台 ExcelExecutor
    -> 普通任务查找图片 / 全局任务持续查找图片
    -> 找到目标后解析 B 列操作指令
    -> DesktopOperations 执行鼠标、键盘或截图操作
    -> 日志反馈执行状态
    -> 到达运行时长或收到停止信号后退出
```

## 四、YOLO 模型能力

YOLOv5 模型代码和 `best.pt` 用于目标检测。当前 `lin_RPA.py` 的用户界面只启动 Excel 线性任务；Excel 中 C 列填写 `全局查找` 时，使用图片模板在整个运行期间进行持续全屏查找，不调用 YOLO 类别监听。

仓库仍保留 `GlobalVisionExecutor` 以及 `J1.py`、`W2.py`、`W3.py` 等 YOLO 实验脚本。它们可以加载 `best.pt`，通过 `mss` 截取屏幕、使用 `torch` 推理，并依据 `data/3leader.yaml` 执行类别动作，但不属于当前 RPA 主界面的启动流程。

## 五、异步和停止机制

后台执行结构如下：

```text
Tkinter 主线程
    +-- 窗口、按钮、日志
    +-- start_async()
          +-- ExcelExecutor
```

`StopController` 内部使用 `threading.Event`：

- 点击“停止”时设置事件。
- 关闭窗口时设置事件并停止键盘监听器。
- Excel 查找、重复点击和循环过程都会检查事件。
- Excel 全局图片查找和普通任务循环都会检查事件。
- 收到信号后线程自然退出，不强制杀死进程。

如果正在执行很长的 `等待=...`，停止会在等待结束后处理。建议把长等待拆成多个短等待。

## 六、配置文件

- `best.pt`：YOLO 训练模型。
- `data/onmyoji.yaml`：模型类别名称，顺序必须和模型类别索引一致。
- `data/onmyoji_name.yaml`：类别名称到中文名称的映射。
- `data/3leader.yaml`：识别类别与动作的映射。
- `settings.json`：程序自动保存的路径、置信度、快捷键等配置；置信度统一从这里读取。

`settings.json` 示例：

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

建议将模板截图放在 `data/images/`，并在 Excel 中使用相对路径，便于项目迁移和打包。

## 七、Python 文件、库和功能

本项目中的 Python 文件并非全部都是独立入口。顶层 RPA 文件负责当前自动化程序，`models/`、`utils/`、`classify/` 和 `segment/` 主要属于 YOLOv5 训练、推理和导出框架。

### 1. RPA 主程序

| 文件 | 主要库 | 实现功能和效果 |
| --- | --- | --- |
| `lin_RPA.py` | `tkinter`、`pathlib`、本项目 RPA 模块 | 创建 Windows 图形界面；选择 Excel 和工作表；设置置信度、运行时间、截图目录；启动和停止后台任务。 |
| `rpa_readers.py` | `json`、`pathlib`、`openpyxl`、`PyYAML` | 读取 `settings.json`、YAML 类别和动作配置；读取 Excel 前四列；解析图片绝对路径和相对路径。 |
| `rpa_operations.py` | `pyautogui`、`pynput`、`numpy`、`threading` | 统一封装鼠标点击、按键、释放、截图、屏幕模板查找和全局快捷键；用 `threading.Event` 传递停止信号。 |
| `rpa_executor.py` | `cv2`、`mss`、`numpy`、`threading`、本项目 RPA 模块 | 执行 Excel 线性任务；支持普通图片查找和第三列填写 `全局查找` 的多个持续图片监听；同时保留 YOLO 类别检测执行器。 |

调用关系如下：

```text
lin_RPA.py
    +-- rpa_readers.py       读取配置和 Excel
    +-- rpa_operations.py    鼠标、截图、停止信号
    +-- rpa_executor.py      组织查找、循环和操作指令
```

### 2. 顶层独立脚本和旧版程序

这些文件保留了项目早期的实验功能。它们通常在导入阶段就会加载模型或创建窗口，使用前应先确认脚本内容和运行环境。

| 文件 | 主要库 | 实现功能和效果 |
| --- | --- | --- |
| `J1.py` | `torch`、`opencv-python`、`mss`、`numpy`、`PyAutoGUI`、`PyYAML`、`tkinter`、`pynput`、`pywin32` | 旧版 YOLO 屏幕检测和透明窗口显示；识别游戏目标后按类别执行点击、随机点击或窗口范围内操作，并显示日志。 |
| `J1-1.py` | `torch`、`opencv-python`、`mss`、`numpy`、`PyAutoGUI`、`PyYAML`、`tkinter`、`pynput`、`pywin32` | `J1.py` 的另一版实验实现，包含 YOLO 持续检测、透明标注窗口、窗口识别和快捷键退出。 |
| `W2.py` | `torch`、`opencv-python`、`mss`、`numpy`、`PyYAML`、`tkinter`、`pynput` | 旧版队长窗口检测程序；截取屏幕后绘制 YOLO 目标和窗口边框，并针对目标类别执行点击。 |
| `W3.py` | `torch`、`opencv-python`、`mss`、`numpy`、`PyYAML`、`tkinter`、`pynput` | `W2.py` 的后续实验版本，继续提供实时目标检测、透明窗口标注和快捷键控制。 |
| `pro.py` | `openpyxl`、`PyAutoGUI`、`numpy`、`pandas`、`tkinter`、`pynput`、标准库 | 早期 Excel 自动化 GUI；按 Excel 行查找图片、等待、点击、截图和跳转，带错误日志、配置保存和快捷键。 |
| `jc.py` | `torch`、`opencv-python`、`mss`、`numpy`、`Pillow`、`PyYAML`、`tkinter` | YOLO 检测结果可视化实验；在透明全屏窗口中绘制检测框、中文类别和置信度。 |
| `屏幕.py` | `tkinter`、`pywin32`、`threading`、`math` | 鼠标跟踪工具；获取鼠标所在窗口的标题、类名和矩形坐标，并用透明窗口实时绘制彩色边框。 |
| `显示文字.py` | `tkinter` | 创建置顶、无边框、透明背景的全屏文字覆盖层，用于在屏幕指定位置显示提示文字。 |
| `benchmarks.py` | `torch`、`pandas`、`numpy`、`psutil`、YOLOv5 `models` 和 `utils` | 对不同模型格式执行速度和精度基准测试，输出推理时间、文件大小和检测结果。 |
| `val.py` | `torch`、`numpy`、`tqdm`、YOLOv5 `models` 和 `utils` | 验证目标检测模型；计算 IoU、Precision、Recall、mAP，生成 JSON/TXT 结果和验证图像。 |
| `export.py` | `torch`、`pandas`、PyTorch Mobile、YOLOv5 `models` 和 `utils` | 将 PyTorch 模型导出为 TorchScript、ONNX、OpenVINO、TensorRT、CoreML、TensorFlow、TFLite 和 PaddlePaddle 等格式。 |
| `hubconf.py` | `torch`、YOLOv5 `models` | 为 `torch.hub.load()` 提供 `custom`、`yolov5s` 等模型加载入口，支持本地或预训练模型。 |



## 八、安装、运行和打包

安装依赖：

```bash
pip install -r requirements.txt
```

如果依赖文件没有列出当前环境所需包，请补充安装 `torch`、`opencv-python`、`mss`、`pyautogui`、`pynput`、`openpyxl`、`PyYAML` 和 `numpy`。

启动：

```bash
python lin_RPA.py
```

打包：

```bash
pyinstaller lin_RPA.spec
```

打包配置会包含 `best.pt`、`settings.json`、`data/*.yaml` 和 Python 模块。生成目录通常为 `dist/lin_RPA/`，移动程序时复制整个目录，不要只复制 exe 文件。




