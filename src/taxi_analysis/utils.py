"""Small serialization utilities shared across analysis stages."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


# JSON이 기본적으로 처리하지 못하는 날짜, 경로, NumPy 값을 일반 Python 값으로 바꾼다.
def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, Path)):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write readable UTF-8 JSON while supporting NumPy and datetime values."""
    # 한글이 이스케이프되지 않도록 UTF-8과 ensure_ascii=False를 사용한다.
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
