"""Persistent CSV storage for real ManGo or Stay assessment records."""
from __future__ import annotations

import csv
import json
import os
import cv2
from datetime import datetime
from pathlib import Path
from typing import Iterable


DEFAULT_HISTORY_PATH = Path("data") / "assessment_history.csv"
FIELDNAMES = [
    "ID",
    "Batch ID",
    "Date",
    "Mango",
    "Ripeness",
    "Confidence",
    "Grade",
    "Defect %",
    "Severity",
    "Defect Types",
    "Processing Info",
    "Original Image Path",
    "Processed Image Path",
    "Defect Overlay Path",
]


def _ensure_history_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=FIELDNAMES).writeheader()
        return

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames == FIELDNAMES:
            return
        existing_rows = list(reader)

    # Migrate older CSV schemas without dropping their assessment data.
    migrated = [_serialize(_deserialize(row)) for row in existing_rows if row.get("ID")]
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(migrated)
    os.replace(temporary_path, path)


def _serialize(record: dict) -> dict:
    date = record.get("Date")
    if isinstance(date, datetime):
        date = date.isoformat(timespec="seconds")
    defect_types = record.get("Defect Types", [])
    if not isinstance(defect_types, list):
        defect_types = [str(defect_types)] if defect_types else []
    return {
        "ID": str(record.get("ID", "")),
        "Batch ID": str(record.get("Batch ID", "Single")),
        "Date": str(date or ""),
        "Mango": str(record.get("Mango", record.get("Filename", ""))),
        "Ripeness": str(record.get("Ripeness", "")),
        "Confidence": float(record.get("Confidence", 0.0)),
        "Grade": str(record.get("Grade", "")),
        "Defect %": float(record.get("Defect %", 0.0)),
        "Severity": str(record.get("Severity", "")),
        "Defect Types": json.dumps(defect_types, ensure_ascii=False),
        "Processing Info": str(record.get("Processing Info", "")),
        "Original Image Path": str(record.get("Original Image Path", "")),
        "Processed Image Path": str(record.get("Processed Image Path", "")),
        "Defect Overlay Path": str(record.get("Defect Overlay Path", "")),
    }


def _deserialize(row: dict) -> dict:
    try:
        defect_types = json.loads(row.get("Defect Types", "[]"))
    except (json.JSONDecodeError, TypeError):
        defect_types = [item.strip() for item in row.get("Defect Types", "").split(",") if item.strip()]
    try:
        date = datetime.fromisoformat(row.get("Date", ""))
    except ValueError:
        date = None
    return {
        "ID": row.get("ID", ""),
        "Batch ID": row.get("Batch ID", "") or "Single",
        "Date": date,
        "Mango": row.get("Mango", ""),
        "Ripeness": row.get("Ripeness", ""),
        "Confidence": float(row.get("Confidence") or 0.0),
        "Grade": row.get("Grade", ""),
        "Defect %": float(row.get("Defect %") or 0.0),
        "Severity": row.get("Severity", ""),
        "Defect Types": defect_types,
        "Processing Info": row.get("Processing Info", ""),
        "Original Image Path": row.get("Original Image Path", ""),
        "Processed Image Path": row.get("Processed Image Path", ""),
        "Defect Overlay Path": row.get("Defect Overlay Path", ""),
    }


def _save_image(image, path: Path, rgb: bool = False) -> str:
    if image is None:
        return ""
    path.parent.mkdir(parents=True, exist_ok=True)
    output = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if rgb else image
    if not cv2.imwrite(str(path), output):
        raise OSError(f"Unable to save assessment image: {path}")
    return str(path.resolve())


def _persist_record_images(record: dict, history_path: Path) -> dict:
    """Save only this record's arrays inside its assessment-ID directory."""
    stored = dict(record)
    assessment_dir = history_path.parent / "assessments" / stored["ID"]
    stored["Original Image Path"] = _save_image(
        stored.get("Original Image"), assessment_dir / "original.jpg"
    ) or stored.get("Original Image Path", "")
    stored["Processed Image Path"] = _save_image(
        stored.get("Processed Image"), assessment_dir / "processed.jpg", rgb=True
    ) or stored.get("Processed Image Path", "")
    stored["Defect Overlay Path"] = _save_image(
        stored.get("Defect Overlay"), assessment_dir / "defect_overlay.jpg"
    ) or stored.get("Defect Overlay Path", "")
    return stored


def load_history(path: str | Path = DEFAULT_HISTORY_PATH) -> list[dict]:
    """Create the CSV if needed and return all stored assessments."""
    history_path = Path(path)
    _ensure_history_file(history_path)
    with history_path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [_deserialize(row) for row in csv.DictReader(handle) if row.get("ID")]


def save_assessments(
    records: Iterable[dict],
    path: str | Path = DEFAULT_HISTORY_PATH,
) -> list[dict]:
    """Persist new records atomically, deduplicated by assessment ID.

    Returns the records that were actually added.
    """
    history_path = Path(path)
    existing = load_history(history_path)
    known_ids = {record["ID"] for record in existing}
    added = []
    for record in records:
        assessment_id = str(record.get("ID", ""))
        if assessment_id and assessment_id not in known_ids:
            added.append(_persist_record_images(record, history_path))
            known_ids.add(assessment_id)
    if not added:
        return []

    serialised = [_serialize(record) for record in [*existing, *added]]
    temporary_path = history_path.with_suffix(history_path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(serialised)
    os.replace(temporary_path, history_path)
    return added


def save_assessment(record: dict, path: str | Path = DEFAULT_HISTORY_PATH) -> bool:
    """Persist one record; return False when its ID already exists."""
    return bool(save_assessments([record], path))
