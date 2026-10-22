"""Load shared settings without importing the simulation or rendering stack."""

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

with (CONFIG_DIR / "exceptions.yaml").open() as file:
    exceptions = yaml.safe_load(file)
    HIGHUP = exceptions["HIGHUP"]
    IGNORED = exceptions["IGNORED"]
    ISOLATED = exceptions["ISOLATED"]

# Explicit broken pairs are currently disabled; isolated categories still apply.
BROKEN = []

with (CONFIG_DIR / "weights.yaml").open() as file:
    weights = yaml.safe_load(file)
    CONSTANTS = weights["CONSTANTS"]
    WEIGHTS = weights["WEIGHTS"]
    CLEARANCE = CONSTANTS["CLEARANCE"]
    SMALL_OFFSET = CONSTANTS["SMALL_OFFSET"]
    LARGE_OFFSET = CONSTANTS["LARGE_OFFSET"]
    MAXIMUN_RETRIES = CONSTANTS["MAXIMUN_RETRIES"]

with (CONFIG_DIR / "robots.yaml").open() as file:
    ROBOTS = yaml.safe_load(file)
