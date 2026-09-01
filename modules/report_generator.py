"""Markdown and PDF report generation for FruitVision assessments."""

from datetime import datetime
from io import BytesIO

import cv2
import numpy as np


def format_coverage(value):
    """Format a validated coverage value for human-readable reports."""

    return "N/A" if value is None else f"{float(value):.2f}%"


def generate_report_string(
    fruit_type,
    batch_id,
    ripeness,
    confidence,
    quality_result,
    severity,
    defect_types,
):
    """Create a Markdown report containing the final grading decision."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    grade = quality_result.get("grade", "Unavailable")
    blemish = format_coverage(quality_result.get("blemish_coverage"))
    damage = format_coverage(quality_result.get("damage_coverage"))
    reason = quality_result.get("reason", "Grading unavailable.")
    return f"""# FruitVision AI - Inspection Report

**Assessment ID:** FR-{datetime.now().strftime('%Y%m%d%H%M%S')}  
**Date & Time:** {timestamp}  
**Batch ID:** {batch_id}  
**Fruit Type:** {fruit_type}

## Analysis Results
* **Predicted Ripeness:** {ripeness if ripeness else "N/A"}
* **Ripeness Confidence:** {confidence:.1f}%
* **Blemish Coverage:** {blemish}
* **Damage Coverage:** {damage}
* **Surface Quality Grade:** {grade}
* **Surface Severity:** {severity}
* **Defect Types Detected:** {', '.join(defect_types) if defect_types else 'None'}
* **Grading Decision:** {reason}

---
*This report was generated automatically by FruitVision AI.*
"""


def _image_png_bytes(image, max_side=1000):
    """Encode an OpenCV image for PDF embedding without temporary files."""

    if image is None:
        return None
    image = np.asarray(image)
    if image.ndim == 2:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 3:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        return None

    height, width = rgb_image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        rgb_image = cv2.resize(
            rgb_image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(
        ".png", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    )
    return bytes(encoded) if ok else None


def generate_pdf_report(
    *,
    fruit_type,
    batch_id,
    ripeness,
    confidence,
    quality_result,
    severity,
    defect_types,
    original_image=None,
    overlay_image=None,
):
    """Return a self-contained PDF report as bytes for Streamlit download."""

    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.lib.utils import ImageReader
        from reportlab.platypus import (
            Image as ReportImage,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise RuntimeError(
            "PDF export requires reportlab. Install it with: "
            "python -m pip install reportlab"
        ) from exc

    grade = quality_result.get("grade", "Unavailable")
    blemish = format_coverage(quality_result.get("blemish_coverage"))
    damage = format_coverage(quality_result.get("damage_coverage"))
    grading_reason = quality_result.get("reason", "Grading unavailable.")
    assessment_id = f"FR-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="FruitVision AI Inspection Report",
        author="FruitVision AI",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#2E7D32"),
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="SmallNote",
        parent=styles["Normal"],
        fontSize=8.5,
        textColor=colors.HexColor("#666666"),
        leading=11,
    ))

    story = [
        Paragraph("FruitVision AI", styles["ReportTitle"]),
        Paragraph("Mango Inspection Report", styles["Heading2"]),
        Spacer(1, 4 * mm),
    ]
    metadata = [
        ["Assessment ID", assessment_id],
        ["Date and time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Batch ID", str(batch_id or "N/A")],
        ["Fruit type", str(fruit_type or "Mango")],
    ]
    metadata_table = Table(metadata, colWidths=[42 * mm, 128 * mm])
    metadata_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8F5E9")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1B5E20")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8E6C9")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([metadata_table, Spacer(1, 7 * mm)])

    story.append(Paragraph("Final result", styles["Heading2"]))
    result_data = [
        ["Ripeness", str(ripeness or "N/A"), f"Confidence: {confidence:.1f}%"],
        ["Blemish coverage", blemish, "Limit: 10%"],
        ["Damage coverage", damage, "Limit: 10%"],
        ["Surface quality", grade, "Rotten always results in Reject"],
        ["Severity", str(severity or "N/A"), ""],
        ["Defect types", ", ".join(defect_types) if defect_types else "None", ""],
    ]
    result_table = Table(result_data, colWidths=[42 * mm, 43 * mm, 85 * mm])
    grade_color = {
        "Premium": "#2E7D32",
        "Grade 1": "#689F38",
        "Grade 2": "#F9A825",
        "Reject": "#C62828",
        "Unavailable": "#757575",
    }.get(grade, "#757575")
    result_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#DDDDDD")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 3), (1, 3), "Helvetica-Bold"),
        ("TEXTCOLOR", (1, 3), (1, 3), colors.HexColor(grade_color)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([result_table, Spacer(1, 5 * mm)])
    story.append(Paragraph(
        f"<b>Grading decision:</b> {grading_reason}",
        styles["Normal"],
    ))
    story.append(Spacer(1, 7 * mm))

    image_cells = []
    for label, image in (("Original image", original_image), ("Defect overlay", overlay_image)):
        encoded = _image_png_bytes(image)
        if encoded:
            image_stream = BytesIO(encoded)
            image_reader = ImageReader(image_stream)
            image_width, image_height = image_reader.getSize()
            display_width = 78 * mm
            display_height = min(
                display_width * image_height / max(image_width, 1),
                85 * mm,
            )
            image_cells.append([
                Paragraph(label, styles["Heading4"]),
                ReportImage(image_stream, width=display_width, height=display_height),
            ])
    if len(image_cells) == 2:
        image_table = Table(
            [[image_cells[0][0], image_cells[1][0]],
             [image_cells[0][1], image_cells[1][1]]],
            colWidths=[85 * mm, 85 * mm],
        )
        image_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(image_table)
    elif image_cells:
        story.append(Table([[image_cells[0][0]], [image_cells[0][1]]]))

    story.extend([
        Spacer(1, 8 * mm),
        Paragraph(
            "The Surface Quality Grade is calculated from the validated "
            "blemish coverage, damage coverage, and ripeness result. "
            "The grading module does not perform additional detection.",
            styles["SmallNote"],
        ),
    ])
    document.build(story)
    return buffer.getvalue()
