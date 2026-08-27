"""线性 Excel 任务与全局视觉监听执行器。"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import mss
import numpy as np

from rpa_operations import Detection, DesktopOperations, StopController, random_position
from rpa_readers import AppConfig, read_excel_tasks, resolve_image_path


class ExcelExecutor:
    def __init__(self, config: AppConfig, operations: DesktopOperations, log: Callable[[str], None] = print):
        self.config = config
        self.operations = operations
        self.log = log
        self.global_executor = GlobalVisionExecutor(config, operations, log)

    def run(self, excel_path: str, sheet_name: str, duration_seconds: int = 0) -> None:
        started = time.monotonic()
        tasks = read_excel_tasks(excel_path, sheet_name)
        global_tasks = []
        for image_name, operation_text, search_timeout, timeout_action in tasks:
            if str(search_timeout).strip() == "全局查找" and image_name and operation_text:
                global_tasks.append((resolve_image_path(str(image_name), self.config.root), operation_text))
        if global_tasks:
            deadline = started + duration_seconds if duration_seconds else None
            start_async(lambda: self._run_global_image_search(global_tasks, deadline))

        row_index = 2
        while not self.operations.stop_controller.stopped():
            if duration_seconds and time.monotonic() - started >= duration_seconds:
                break
            task_index = (row_index - 2) % max(1, len(tasks))
            if not tasks:
                self.log("Excel 没有可执行任务")
                break
            image_name, operation_text, search_timeout, timeout_action = tasks[task_index]
            row_index += 1
            if not image_name or not operation_text:
                continue
            operation_text = str(operation_text)
            if str(search_timeout).strip() == "全局查找":
                continue
            timeout = float(search_timeout or 10)
            image_path = resolve_image_path(str(image_name), self.config.root)
            location = self._wait_for_image(image_path, timeout)
            if location is None:
                self.log(f"图片查找超时: {image_name}")
                row_index = self._timeout_row(row_index, timeout_action)
                continue
            self._run_operations(operation_text, location)

    def _run_global_image_search(self, tasks, deadline: float | None = None) -> None:
        active_paths = set()
        while not self.operations.stop_controller.stopped() and (deadline is None or time.monotonic() < deadline):
            for image_path, operation_text in tasks:
                try:
                    location = self.operations.locate(image_path, self.config.confidence)
                except Exception:
                    continue

                if location:
                    if image_path not in active_paths:
                        self._run_operations(operation_text, location)
                        active_paths.add(image_path)
                else:
                    active_paths.discard(image_path)
            time.sleep(0.1)

    def _wait_for_image(self, image_path: Path, timeout: float):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self.operations.stop_controller.stopped():
            try:
                location = self.operations.locate(image_path, self.config.confidence)
                if location:
                    self.log(f"找到图片: {image_path}")
                    return location
            except Exception as error:
                self.log(f"查找图片失败: {error}")
            time.sleep(0.1)
        return None

    def _timeout_row(self, current_row: int, action: Any) -> int:
        if str(action).isdigit():
            return int(action)
        return current_row

    def _run_operations(self, operation_text: str, location) -> None:
        x, y = location.left + location.width / 2, location.top + location.height / 2
        for operation in (item.strip() for item in str(operation_text).split("，")):
            if self.operations.stop_controller.stopped():
                return
            if operation == "全局查找":
                continue
            if operation.startswith("等待="):
                time.sleep(float(operation.split("=", 1)[1]))
            elif operation == "左键按下":
                self.operations.press("left")
            elif operation == "左键释放":
                self.operations.release("left")
            elif operation == "右键按下":
                self.operations.press("right")
            elif operation == "右键释放":
                self.operations.release("right")
            elif operation.startswith("偏移="):
                dx, dy = map(int, operation.split("=", 1)[1].split("/"))
                x, y = x + dx, y + dy
            elif operation.startswith("左键="):
                self._click_many(x, y, int(operation.split("=", 1)[1]))
            elif operation.startswith("二级左键="):
                count = int(operation.split("=", 1)[1])
                for _ in range(count):
                    self.operations.click(random_position(x, location.left, location.left + location.width), random_position(y, location.top, location.top + location.height))
            elif operation.startswith("三级左键="):
                dx, dy, count = map(int, operation.split("=", 1)[1].split("/"))
                self._click_many(x + dx, y + dy, count)
            elif operation == "屏幕截图":
                self.operations.screenshot(self.config.screenshot_path or self.config.root / "output")
            elif operation.startswith("区域屏幕截图="):
                region = tuple(map(int, operation.split("=", 1)[1].split("/")))
                self.operations.screenshot(self.config.screenshot_path or self.config.root / "output", region=region)
            else:
                self.log(f"未知操作: {operation}")

    def _click_many(self, x: float, y: float, count: int) -> None:
        for _ in range(count):
            if self.operations.stop_controller.stopped():
                return
            self.operations.click(x, y)


class GlobalVisionExecutor:
    def __init__(self, config: AppConfig, operations: DesktopOperations, log: Callable[[str], None] = print):
        self.config = config
        self.operations = operations
        self.log = log
        self.model = None

    def load_model(self) -> None:
        import torch
        model_path = self.config.root / self.config.model_path
        if not model_path.exists():
            raise FileNotFoundError(f"模型不存在: {model_path}")
        self.model = torch.hub.load(str(self.config.root), "custom", path=str(model_path), source="local")

    def run(self, monitor: dict[str, int] | None = None) -> None:
        if self.model is None:
            self.load_model()
        monitor = monitor or {"top": 0, "left": 0, "width": 1920, "height": 1080}
        with mss.mss() as screen:
            while not self.operations.stop_controller.stopped():
                detections = self.detect(screen, monitor)
                for detection in detections:
                    name = detection.name
                    action = (self.config.actions or {}).get(name)
                    if action and not action.get("special"):
                        self.operations.click(detection.center_x, detection.center_y)
                        self.log(f"模型监听操作: {name}")

    def detect(self, screen, monitor: dict[str, int]) -> list[Detection]:
        frame = cv2.cvtColor(np.array(screen.grab(monitor)), cv2.COLOR_BGRA2RGB)
        results = self.model(frame)
        detections = []
        for *box, confidence, class_index in results.xyxy[0].cpu().numpy():
            if confidence < self.config.confidence:
                continue
            x1, y1, x2, y2 = (int(value) for value in box)
            x1 += monitor["left"]
            x2 += monitor["left"]
            y1 += monitor["top"]
            y2 += monitor["top"]
            detections.append(Detection(
                name=self.config.class_names[int(class_index)],
                confidence=float(confidence),
                center_x=(x1 + x2) // 2,
                center_y=(y1 + y2) // 2,
                box=(x1, y1, x2, y2),
            ))
        return detections

    def wait_for_class(self, target_name: str, timeout: float) -> Detection | None:
        if self.model is None:
            self.load_model()
        deadline = time.monotonic() + timeout
        monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        with mss.mss() as screen:
            while time.monotonic() < deadline and not self.operations.stop_controller.stopped():
                for detection in self.detect(screen, monitor):
                    if detection.name == target_name:
                        return detection
                time.sleep(0.1)
        return None


def start_async(target: Callable[[], None]) -> threading.Thread:
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread
