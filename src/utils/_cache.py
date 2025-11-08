from pathlib import Path

from ..config import get_cache_dir, get_project_root


def ensure_cache_dir() -> Path:
    env_cache_str = get_cache_dir()
    env_cache_path = Path(env_cache_str)
    if env_cache_path.is_absolute():
        cache_dir = env_cache_path
    else:
        project_root = get_project_root()
        cache_dir = (project_root / env_cache_path).resolve()
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_path(filename: str | None = None) -> Path:
    cache_dir = ensure_cache_dir()
    if filename is None:
        return cache_dir
    return cache_dir / filename
