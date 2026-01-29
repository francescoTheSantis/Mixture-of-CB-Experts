from os import environ as env
from pathlib import Path
import os

PROJECT_NAME = "" # Name of the project
HOME = "" # Path to the project

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

DATA_PATH = "" # Path to the datasets

# OpenAI API key for data augmentation
OPENAI_API_KEY = ""

env['HYDRA_FULL_ERROR'] = '1'

