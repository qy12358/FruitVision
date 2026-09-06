"""Durable storage for completed FruitVision assessments.

Images are intentionally not stored here.  The database contains the
structured measurements and decisions needed for history, filtering, and
future reporting while PDF exports remain download-only.
"""

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "data" / "fruitvision_assessments.db"


def _connect(database_path=DATABASE_PATH):
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    return connection


def initialise_store(database_path=DATABASE_PATH):
    """Create the assessment table if it does not exist yet."""

    with closing(_connect(database_path)) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS assessments (
                id TEXT PRIMARY KEY,
                fruit_type TEXT NOT NULL,
                batch_id TEXT,
                ripeness TEXT,
                confidence REAL,
                blemish_coverage REAL,
                damage_coverage REAL,
                defect_coverage REAL,
                grade TEXT NOT NULL,
                interpretation TEXT,
                severity TEXT,
                defect_types TEXT,
                decision_source TEXT,
                analysis_json TEXT,
                assessed_at TEXT NOT NULL
            )
            """
        )
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(assessments)")
        }
        if "analysis_json" not in columns:
            connection.execute(
                "ALTER TABLE assessments ADD COLUMN analysis_json TEXT"
            )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS assessment_artifacts (
                assessment_id TEXT NOT NULL,
                artifact_name TEXT NOT NULL,
                artifact_format TEXT NOT NULL,
                artifact_data BLOB NOT NULL,
                PRIMARY KEY (assessment_id, artifact_name),
                FOREIGN KEY (assessment_id) REFERENCES assessments(id)
            )
            """
        )
        connection.commit()


def _json_safe(value, artifacts, path="data", seen=None):
    """Convert analysis output to JSON while storing arrays as artifacts."""

    if seen is None:
        seen = {}
    if isinstance(value, np.ndarray):
        existing_name = seen.get(id(value))
        if existing_name:
            return {"__artifact__": existing_name}
        artifact_name = path.replace(" ", "_")
        seen[id(value)] = artifact_name
        buffer = BytesIO()
        np.save(buffer, value, allow_pickle=False)
        artifacts[artifact_name] = ("npy", buffer.getvalue())
        return {
            "__artifact__": artifact_name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item, artifacts, f"{path}_{key}", seen)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, artifacts, f"{path}_{index}", seen)
            for index, item in enumerate(value)
        ]
    if isinstance(value, (set, frozenset)):
        return [
            _json_safe(item, artifacts, f"{path}_{index}", seen)
            for index, item in enumerate(sorted(value, key=str))
        ]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _restore_artifacts(value, artifacts):
    """Replace artifact references in loaded JSON with NumPy arrays."""

    if isinstance(value, dict):
        if "__artifact__" in value:
            return artifacts.get(value["__artifact__"])
        return {
            key: _restore_artifacts(item, artifacts)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_restore_artifacts(item, artifacts) for item in value]
    return value


def _row_to_assessment(row):
    """Map a database row to the application's display field names."""

    item = dict(row)
    item["ID"] = item.pop("id")
    item["Fruit Type"] = item.pop("fruit_type")
    item["Batch ID"] = item.pop("batch_id") or ""
    item["Ripeness"] = item.pop("ripeness") or ""
    item["Confidence"] = item.pop("confidence")
    item["Blemish %"] = item.pop("blemish_coverage")
    item["Damage %"] = item.pop("damage_coverage")
    item["Defect %"] = item.pop("defect_coverage")
    item["Grade"] = item.pop("grade")
    item["Interpretation"] = item.pop("interpretation") or ""
    item["Severity"] = item.pop("severity") or ""
    try:
        item["Defect Types"] = json.loads(item.pop("defect_types") or "[]")
    except json.JSONDecodeError:
        item["Defect Types"] = []
    item["Decision Source"] = item.pop("decision_source") or ""
    analysis_json = item.pop("analysis_json", None)
    try:
        item["Analysis Data"] = json.loads(analysis_json) if analysis_json else {}
    except json.JSONDecodeError:
        item["Analysis Data"] = {}
    item["Date"] = datetime.fromisoformat(item.pop("assessed_at"))
    return item


def save_assessment(assessment, database_path=DATABASE_PATH):
    """Insert or replace one completed assessment and return its ID."""

    initialise_store(database_path)
    artifacts = {}
    analysis_json = json.dumps(
        _json_safe(assessment.get("Analysis Data", {}), artifacts)
    )
    assessed_at = assessment.get("Date", datetime.now())
    if isinstance(assessed_at, datetime):
        assessed_at = assessed_at.isoformat(timespec="seconds")

    with closing(_connect(database_path)) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO assessments (
                id, fruit_type, batch_id, ripeness, confidence,
                blemish_coverage, damage_coverage, defect_coverage, grade,
                interpretation, severity, defect_types, decision_source,
                analysis_json, assessed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                assessment["ID"],
                assessment.get("Fruit Type", "Mango"),
                assessment.get("Batch ID", ""),
                assessment.get("Ripeness", ""),
                assessment.get("Confidence"),
                assessment.get("Blemish %"),
                assessment.get("Damage %"),
                assessment.get("Defect %"),
                assessment.get("Grade", "Unavailable"),
                assessment.get("Interpretation", ""),
                assessment.get("Severity", ""),
                json.dumps(assessment.get("Defect Types", [])),
                assessment.get("Decision Source", ""),
                analysis_json,
                assessed_at,
            ),
        )
        connection.execute(
            "DELETE FROM assessment_artifacts WHERE assessment_id = ?",
            (assessment["ID"],),
        )
        connection.executemany(
            """
            INSERT INTO assessment_artifacts
                (assessment_id, artifact_name, artifact_format, artifact_data)
            VALUES (?, ?, ?, ?)
            """,
            [
                (assessment["ID"], name, file_format, data)
                for name, (file_format, data) in artifacts.items()
            ],
        )
        connection.commit()
    return assessment["ID"]


def load_assessments(database_path=DATABASE_PATH):
    """Return saved assessments as dictionaries, newest first."""

    initialise_store(database_path)
    with closing(_connect(database_path)) as connection:
        rows = connection.execute(
            "SELECT * FROM assessments ORDER BY assessed_at DESC"
        ).fetchall()

    return [_row_to_assessment(row) for row in rows]


def load_assessment(assessment_id, database_path=DATABASE_PATH):
    """Load one assessment and restore all stored image artifacts."""

    initialise_store(database_path)
    with closing(_connect(database_path)) as connection:
        row = connection.execute(
            "SELECT * FROM assessments WHERE id = ?",
            (assessment_id,),
        ).fetchone()
        if row is None:
            return None
        item = _row_to_assessment(row)
        artifact_rows = connection.execute(
            """
            SELECT artifact_name, artifact_format, artifact_data
            FROM assessment_artifacts
            WHERE assessment_id = ?
            """,
            (assessment_id,),
        ).fetchall()

    artifacts = {}
    for artifact_row in artifact_rows:
        if artifact_row["artifact_format"] != "npy":
            continue
        artifacts[artifact_row["artifact_name"]] = np.load(
            BytesIO(artifact_row["artifact_data"]),
            allow_pickle=False,
        )
    item["Analysis Data"] = _restore_artifacts(
        item.get("Analysis Data", {}), artifacts
    )
    return item
