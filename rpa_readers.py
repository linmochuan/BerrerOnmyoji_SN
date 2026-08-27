"""读取 RPA 配置、Excel 任务和模型类别。"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import openpyxl
import yaml


@dataclass(frozen=True)
class AppConfig:
    root: Path
    confidence: float = 0.8
    excel_path: str = ""
    sheet_name: str = ""
    screenshot_path: str = ""
    hotkey: tuple[str, ...] = ()
    model_path: str = "best.pt"
    class_names: tuple[str, ...] = ()
    class_name_to_chinese: dict[str, str] | None = None
    actions: dict[str, dict[str, Any]] | None = None


def project_root() -> Path:
    if getattr(__import__("sys"), "frozen", False):
        return Path(getattr(__import__("sys"), "_MEIPASS"))
    return Path(__file__).resolve().parent


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def read_app_config(root: Path | None = None, settings_path: Path | None = None) -> AppConfig:
    root = root or project_root()
    settings_path = settings_path or root / "settings.json"
    settings: dict[str, Any] = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            settings = {}

    try:
        confidence = float(settings.get("confidence", 0.8))
    except (TypeError, ValueError):
        confidence = 0.8
    confidence = min(1.0, max(0.1, confidence))

    names = _load_yaml(root / "data" / "onmyoji.yaml").get("names", [])
    if isinstance(names, dict):
        names = [names[index] for index in sorted(names, key=lambda value: int(value))]
    translations = _load_yaml(root / "data" / "onmyoji_name.yaml").get(
        "Class_Name_To_Chinese", {}
    )
    actions = _load_yaml(root / "data" / "3leader.yaml").get("actions", {})
    hotkey = tuple(str(value) for value in settings.get("hotkey", []))
    return AppConfig(
        root=root,
        confidence=confidence,
        excel_path=str(settings.get("excel_path", "")),
        screenshot_path=str(settings.get("screenshot_path", "")),
        hotkey=hotkey,
        model_path=str(settings.get("model_path", "best.pt")),
        class_names=tuple(str(name) for name in names),
        class_name_to_chinese=translations,
        actions=actions,
    )


def save_settings(config: AppConfig, path: Path | None = None) -> None:
    path = path or config.root / "settings.json"
    data = {
        "confidence": config.confidence,
        "excel_path": config.excel_path,
        "sheet_name": config.sheet_name,
        "screenshot_path": config.screenshot_path,
        "hotkey": list(config.hotkey),
        "model_path": config.model_path,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")


def read_excel_tasks(path: str | Path, sheet_name: str) -> list[tuple[Any, Any, Any, Any]]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        if sheet_name not in workbook.sheetnames:
            raise ValueError(f"工作表不存在: {sheet_name}")
        sheet = workbook[sheet_name]
        return [tuple(row) for row in sheet.iter_rows(min_row=2, max_col=4, values_only=True)]
    finally:
        workbook.close()


def resolve_image_path(image_name: str, root: Path) -> Path:
    candidate = Path(os.path.expandvars(os.path.expanduser(image_name)))
    if candidate.is_absolute():
        return candidate
    return root / candidate
