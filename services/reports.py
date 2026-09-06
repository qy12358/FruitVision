"""services / reports for ManGo or Stay."""

from config import FRUIT_TYPE
from modules.report_generator import (
    deserialize_report_details,
    format_coverage,
    generate_pdf_report,
    generate_report_string,
)
from services.storage import decode_png
import cv2

def report_data_from_record(record: dict):
    original = decode_png(record.get("original_image"), cv2.IMREAD_COLOR)
    overlay = decode_png(record.get("overlay_image"), cv2.IMREAD_COLOR)
    details = deserialize_report_details(record.get("report_details_json"))
    if not details:
        # Older records retain only a subset of the intermediate images.
        details = {key: record.get(key) for key in (
            'probabilities', 'hsv_features', 'statistical_analysis', 'preprocessing_time',
            'inference_time_ms', 'feature_count', 'mango_pixel_count', 'total_pixel_count', 'accepted_count')}
        details['images'] = {key: decode_png(record.get(column), cv2.IMREAD_UNCHANGED)
                             for key, column in (('segmented_bgr', 'processed_image'),
                                                 ('fruit_mask', 'fruit_mask'),
                                                 ('damage_mask', 'damage_mask'))}
    report_text = generate_report_string(
        fruit_type=record.get("fruit_type", FRUIT_TYPE),
        batch_id=record.get("batch_id", ""),
        ripeness=record.get("ripeness"),
        confidence=float(record.get("confidence") or 0.0),
        quality_result=record.get("quality_result", {}),
        severity=record.get("severity"),
        defect_types=record.get("defect_types", []),
        report_details=details,
    )
    try:
        pdf = generate_pdf_report(
            fruit_type=record.get("fruit_type", FRUIT_TYPE),
            batch_id=record.get("batch_id", ""),
            ripeness=record.get("ripeness"),
            confidence=float(record.get("confidence") or 0.0),
            quality_result=record.get("quality_result", {}),
            severity=record.get("severity"),
            defect_types=record.get("defect_types", []),
            original_image=original,
            overlay_image=overlay,
            report_details=details,
        )
    except (RuntimeError, ValueError) as exc:
        pdf = None
        return report_text, pdf, str(exc)
    return report_text, pdf, None

