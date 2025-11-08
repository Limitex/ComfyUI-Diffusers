from ._env import get_str


def get_cache_dir() -> str:
    cache_dir = get_str("COMFYUI_DIFFUSERS_CACHE_DIR", "./cache")
    if cache_dir is None:
        raise ValueError("Cache directory environment variable is not set.")
    return cache_dir
