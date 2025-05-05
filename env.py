from os import environ as env
from pathlib import Path
import os

PROJECT_NAME = 'linear_memory_reasoner' # Name of the project
HOME = "/home/fdesantis/projects/Linear-Memory-Reasoner" # Path to the project

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

DATASETS = os.path.join(str(CACHE), 'datasets')
os.makedirs(DATASETS, exist_ok=True)

env['HYDRA_FULL_ERROR'] = '1'

