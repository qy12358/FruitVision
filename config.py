"""config for ManGo or Stay."""

from pathlib import Path

PRIMARY = "#244B3E"

PRIMARY_LIGHT = "#789184"

SECONDARY = "#B9854D"

BG_LIGHT = "#F6F5F1"

RED = "#A75454"

YELLOW = "#B78C49"

GREEN = "#4F7863"

PAGES = ["Home", "Assess Mango", "Assessment Details", "History & Reports"]

FRUIT_TYPE = "Harumanis Mango"

RIPENESS_ADVICE = {
    "Ripe": "This mango appears ready to eat or sell now.",
    "Semi-Ripe": "This mango may need a little more time before it is fully ripe.",
    "Unripe": "This mango needs more time to ripen.",
    "Rotten": "This mango shows signs of deterioration and should be checked before use.",
}

RIPENESS_BADGE_CLASS = {
    "Unripe": "badge-unripe",
    "Semi-Ripe": "badge-semi",
    "Ripe": "badge-ripe",
    "Rotten": "badge-rotten",
}

RIPENESS_COLORS = {
    "Unripe": SECONDARY,
    "Semi-Ripe": YELLOW,
    "Ripe": GREEN,
    "Rotten": RED,
}

APP_DIR = Path(__file__).resolve().parent

DATA_DIR = APP_DIR / "data"

DB_PATH = DATA_DIR / "mango_assessments.db"

