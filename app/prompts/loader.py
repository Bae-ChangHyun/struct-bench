from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

TEMPLATES_DIR = Path(__file__).parent / "templates"


class PromptTemplate:
    def __init__(self, data: dict):
        self.name: str = data["name"]
        self.description: str = data.get("description", "")
        self.version: str = data.get("version", "1.0")
        self.system_prompt: str = data["system_prompt"]


@lru_cache(maxsize=None)
def load_prompt(name: str) -> PromptTemplate:
    path = TEMPLATES_DIR / f"{name}.yaml"
    if not path.exists():
        available = [p.stem for p in TEMPLATES_DIR.glob("*.yaml")]
        raise FileNotFoundError(
            f"Prompt '{name}' not found. Available: {', '.join(available)}"
        )
    with open(path) as f:
        data = yaml.safe_load(f)
    return PromptTemplate(data)


def list_prompts() -> list[str]:
    return sorted(p.stem for p in TEMPLATES_DIR.glob("*.yaml"))
