"""Repository path constants (demo data, config, runtime outputs)."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = REPO_ROOT / "config"
DATA_DEMO_DIR = REPO_ROOT / "data" / "demo"
DEFAULT_DEMO_CSV = DATA_DEMO_DIR / "2020_03_13_chishuru0016_4_angles_data.csv"
DEFAULT_DEMO_CSV_ALT = DATA_DEMO_DIR / "2020_03_13_chishuru0003_4_angles_data.csv"
DEFAULT_ARM_AXES_CONFIG = CONFIG_DIR / "soft_arm_arm_axes.json"
DEFAULT_COORD_FILE = REPO_ROOT / "ball_target.json"
OUTPUT_V2_SIM_DIR = REPO_ROOT / "output_v2_sim"
OUTPUT_V3_SIM_DIR = REPO_ROOT / "output_v3_sim"
