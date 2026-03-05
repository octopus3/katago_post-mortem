"""测试配置 — 确保 backend 目录在导入路径中，提供公共 fixture。"""

import sys
from pathlib import Path

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


@pytest.fixture(autouse=True)
def _clean_game_sessions():
    """每个测试运行前后清空 main 模块中的 _games 字典，避免状态泄漏。"""
    import main
    main._games.clear()
    yield
    main._games.clear()
