"""桌面输入、截图、模板匹配和全局热键操作。"""
from dataclasses import dataclass
import random
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pyautogui
from pynput import keyboard


class StopController:
    def __init__(self) -> None:
        self.event = threading.Event()

    def stop(self) -> None:
        self.event.set()

    def reset(self) -> None:
        self.event.clear()

    def stopped(self) -> bool:
        return self.event.is_set()


class DesktopOperations:
    def __init__(self, stop_controller: StopController, log: Callable[[str], None] = print):
        self.stop_controller = stop_controller
        self.log = log

    def click(self, x: float, y: float) -> None:
        pyautogui.click(x, y, duration=random.uniform(0.064, 0.1))

    def press(self, button: str = "left") -> None:
        pyautogui.mouseDown(button=button)

    def release(self, button: str = "left") -> None:
        pyautogui.mouseUp(button=button)

    def locate(self, image_path: str | Path, confidence: float):
        return pyautogui.locateOnScreen(str(image_path), confidence=confidence)

    def screenshot(self, directory: str | Path, region: tuple[int, int, int, int] | None = None) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / f"screenshot_{time.strftime('%Y%m%d_%H%M%S_%f')}.png"
        pyautogui.screenshot(str(output), region=region)
        return output

    def start_hotkey_listener(self, hotkey: set[str], on_stop: Callable[[], None]) -> keyboard.Listener:
        pressed: set[str] = set()

        def key_name(key) -> str:
            if getattr(key, "char", None):
                return key.char.upper()
            return str(key).replace("Key.", "").upper()

        def on_press(key) -> None:
            pressed.add(key_name(key))
            if hotkey and hotkey.issubset(pressed):
                on_stop()

        def on_release(key) -> None:
            pressed.discard(key_name(key))

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.daemon = True
        listener.start()
        return listener


def random_position(center: float, minimum: float, maximum: float) -> float:
    deviation = (maximum - minimum) / 6.0
    return float(np.clip(np.random.normal(center, deviation), minimum, maximum))


@dataclass(frozen=True)
class Detection:
    name: str
    confidence: float
    center_x: int
    center_y: int
    box: tuple[int, int, int, int]

    @property
    def left(self) -> int:
        return self.box[0]

    @property
    def top(self) -> int:
        return self.box[1]

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]
