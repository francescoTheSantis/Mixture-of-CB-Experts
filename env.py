from os import environ as env
from pathlib import Path
import os

PROJECT_NAME = 'linear_memory_reasoner'

CACHE = Path(
    env.get(
        f"{PROJECT_NAME.upper()}_CACHE",
        Path(
            env.get("XDG_CACHE_HOME", Path("~", ".cache")),
            PROJECT_NAME,
        ),
    )
).expanduser()
CACHE.mkdir(exist_ok=True)

env['HYDRA_FULL_ERROR'] = '1'

