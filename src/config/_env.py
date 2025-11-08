from __future__ import annotations

import os
from functools import lru_cache
from typing import Final

from dotenv import find_dotenv, load_dotenv

from ._path import get_project_root

_TRUE_SET: Final[set[str]] = {"1", "true", "yes", "on"}
_FALSE_SET: Final[set[str]] = {"0", "false", "no", "off"}


@lru_cache(maxsize=1)
def _load_envs_once() -> None:
    root = get_project_root()
    dotenv_path = str(root / ".env")
    default_dotenv_path = str(root / ".env.default")
    load_dotenv(find_dotenv(dotenv_path, usecwd=True), override=False)
    load_dotenv(find_dotenv(default_dotenv_path, usecwd=True), override=False)


def load_envs(force: bool = False) -> None:
    if force:
        _load_envs_once.cache_clear()
    _load_envs_once()


def get_str(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


def get_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE_SET:
        return True
    if value in _FALSE_SET:
        return False
    raise ValueError(f"Environment variable '{name}' has invalid boolean value '{raw}'")


def get_int(name: str, default: int | None = None) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"Environment variable '{name}' has invalid integer value '{raw}'"
        ) from err


def require(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise KeyError(f"Required environment variable not set: '{name}'")
    return value
