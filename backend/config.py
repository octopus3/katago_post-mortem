"""配置管理模块 — 读取并校验 config.yaml"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Union

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


@dataclass
class KataGoConfig:
    executable: str = ""
    model: str = ""
    config: str = ""
    max_visits: int = 500
    threads: int = 4


@dataclass
class ReviewConfig:
    winrate_threshold: float = 0.05
    top_variations: int = 3
    variation_depth: int = 8


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "MiniMax-M2.5"
    api_key: str = ""
    base_url: str = "https://api.minimaxi.com/v1"


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class AppConfig:
    katago: KataGoConfig = field(default_factory=KataGoConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


def load_config(path: Optional[Union[Path, str]] = None) -> AppConfig:
    """从 YAML 文件加载配置，缺失字段使用默认值。"""
    path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f) or {}

    return AppConfig(
        katago=KataGoConfig(**raw.get("katago", {})),
        review=ReviewConfig(**raw.get("review", {})),
        llm=LLMConfig(**raw.get("llm", {})),
        server=ServerConfig(**raw.get("server", {})),
    )
