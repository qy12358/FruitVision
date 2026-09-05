"""services / storage for ManGo or Stay."""

from config import DATA_DIR
from config import DB_PATH
from config import FRUIT_TYPE
from datetime import datetime
import cv2
import json
import numpy as np
import pandas as pd
import sqlite3

def initialise_database():
    """Create the local assessment database if it does not already exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS assessments (
                assessment_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                fruit_type TEXT NOT NULL,
                batch_id TEXT,
                ripeness TEXT,
                confidence REAL,
                grade TEXT,
                defect_percentage REAL,
                blemish_percentage REAL,
                damage_percentage REAL,
                severity TEXT,
                defect_types_json TEXT,
                quality_result_json TEXT,
                probabilities_json TEXT,
                hsv_features_json TEXT,
                statistical_analysis_json TEXT,
                colour_histogram_json TEXT,
                feature_count INTEGER,
                inference_time_ms REAL,
                preprocessing_time REAL,
                contrast_method TEXT,
                mango_pixel_count INTEGER,
                total_pixel_count INTEGER,
                accepted_count INTEGER,
                rejected_count INTEGER,
                original_image BLOB,
                processed_image BLOB,
                overlay_image BLOB,
                damage_mask BLOB,
                blackhat_image BLOB,
                fruit_mask BLOB
            )
            """
        )
        connection.commit()

def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")

def to_json_text(value) -> str:
    return json.dumps(value, default=json_default, ensure_ascii=False)

def from_json_text(value, fallback):
    if value in (None, ""):
        return fallback
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback

def encode_png(image: np.ndarray | None, colour_order: str = "BGR") -> bytes | None:
    """Encode an OpenCV/Numpy image as PNG bytes for durable storage."""
    if image is None:
        return None
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 3 and colour_order.upper() == "RGB":
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".png", array)
    if not success:
        raise ValueError("The assessment image could not be encoded for storage.")
    return buffer.tobytes()

def decode_png(blob: bytes | None, flags=cv2.IMREAD_COLOR) -> np.ndarray | None:
    if not blob:
        return None
    array = np.frombuffer(blob, dtype=np.uint8)
    return cv2.imdecode(array, flags)

def save_assessment(analysis: dict, batch_id: str) -> str:
    """Persist one completed, real assessment and all data needed to review it later."""
    assessment_id = f"MS-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    result = analysis["result"]
    blemish_result = analysis["blemish_result"]

    values = (
        assessment_id,
        datetime.now().isoformat(timespec="seconds"),
        FRUIT_TYPE,
        batch_id.strip(),
        analysis.get("ripeness"),
        float(analysis.get("confidence", 0.0)),
        analysis.get("grade"),
        float(analysis.get("defect_pct", 0.0)),
        float(analysis.get("blemish_pct", 0.0)),
        float(analysis.get("damage_pct", 0.0)),
        analysis.get("severity"),
        to_json_text(analysis.get("defect_types", [])),
        to_json_text(analysis.get("quality_result", {})),
        to_json_text(analysis.get("probabilities", {})),
        to_json_text(analysis.get("hsv_features", {})),
        to_json_text(analysis.get("statistical_analysis", {})),
        to_json_text(analysis.get("colour_histogram", {})),
        int(analysis.get("feature_count", 0)),
        float(analysis.get("inference_time_ms", 0.0)),
        float(analysis.get("preprocessing_time", 0.0)),
        str(result.get("contrast_method", "Unavailable")),
        int(analysis.get("mango_pixel_count", 0)),
        int(analysis.get("total_pixel_count", 0)),
        int(analysis.get("accepted_count", 0)),
        int(analysis.get("rejected_count", 0)),
        sqlite3.Binary(encode_png(analysis.get("image"), "BGR")),
        sqlite3.Binary(encode_png(result.get("segmented_rgb"), "RGB")),
        sqlite3.Binary(encode_png(blemish_result.get("overlay"), "BGR")),
        sqlite3.Binary(encode_png(blemish_result.get("damage_mask"), "BGR")),
        sqlite3.Binary(encode_png(blemish_result.get("blackhat"), "BGR")),
        sqlite3.Binary(encode_png(result.get("fruit_mask"), "BGR")),
    )

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO assessments (
                assessment_id, created_at, fruit_type, batch_id, ripeness, confidence,
                grade, defect_percentage, blemish_percentage, damage_percentage,
                severity, defect_types_json, quality_result_json, probabilities_json,
                hsv_features_json, statistical_analysis_json, colour_histogram_json,
                feature_count, inference_time_ms, preprocessing_time, contrast_method,
                mango_pixel_count, total_pixel_count, accepted_count, rejected_count,
                original_image, processed_image, overlay_image, damage_mask,
                blackhat_image, fruit_mask
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )
        connection.commit()
    return assessment_id

def load_history_dataframe() -> pd.DataFrame:
    """Load only summary fields used by Home and History pages."""
    query = """
        SELECT assessment_id, created_at, fruit_type, batch_id, ripeness, confidence,
               grade, defect_percentage, blemish_percentage, damage_percentage,
               severity, defect_types_json, inference_time_ms
        FROM assessments
        ORDER BY created_at DESC
    """
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(query).fetchall()

    columns = [
        "ID", "Created At", "Fruit Type", "Batch ID", "Ripeness", "Confidence",
        "Grade", "Defect %", "Blemish %", "Damage %", "Severity",
        "Defect Types JSON", "Inference ms",
    ]
    history = pd.DataFrame(rows, columns=columns)
    if history.empty:
        history["Date"] = pd.to_datetime(pd.Series(dtype="datetime64[ns]"))
        history["Defect Types"] = pd.Series(dtype="object")
        return history

    history["Date"] = pd.to_datetime(history["Created At"], errors="coerce")
    history["Defect Types"] = history["Defect Types JSON"].apply(
        lambda value: from_json_text(value, [])
    )
    return history

def fetch_assessment(assessment_id: str) -> dict | None:
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM assessments WHERE assessment_id = ?",
            (assessment_id,),
        ).fetchone()
    if row is None:
        return None

    record = dict(row)
    for source, target, fallback in (
        ("defect_types_json", "defect_types", []),
        ("quality_result_json", "quality_result", {}),
        ("probabilities_json", "probabilities", {}),
        ("hsv_features_json", "hsv_features", {}),
        ("statistical_analysis_json", "statistical_analysis", {}),
        ("colour_histogram_json", "colour_histogram", {}),
    ):
        record[target] = from_json_text(record.get(source), fallback)
    record["created_at_dt"] = pd.to_datetime(record.get("created_at"), errors="coerce")
    return record

