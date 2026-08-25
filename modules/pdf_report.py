"""PDF reporting for ManGo or Stay assessments."""
from __future__ import annotations

from io import BytesIO
import cv2
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak


def _image_flowable(array=None, bgr=False, path=None, width=72 * mm, height=72 * mm):
    if array is None and path:
        image_path = Path(path)
        if image_path.is_file():
            array = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            bgr = True
    if array is None:
        return None
    converted = cv2.cvtColor(array, cv2.COLOR_BGR2RGB) if bgr else array
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(converted, cv2.COLOR_RGB2BGR))
    if not ok:
        return None
    return Image(BytesIO(encoded.tobytes()), width=width, height=height, kind="proportional")


def _assessment_images(assessment, width=72 * mm, height=72 * mm):
    return [
        ("Original mango image", _image_flowable(
            assessment.get("Original Image"), True,
            assessment.get("Original Image Path"), width, height)),
        ("Segmented / processed image", _image_flowable(
            assessment.get("Processed Image"), False,
            assessment.get("Processed Image Path"), width, height)),
        ("Defect overlay", _image_flowable(
            assessment.get("Defect Overlay"), True,
            assessment.get("Defect Overlay Path"), width, height)),
    ]


def generate_assessment_pdf(assessment: dict) -> bytes:
    """Return genuine PDF bytes; no filesystem or model files are modified."""
    output = BytesIO()
    doc = SimpleDocTemplate(output, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
                            topMargin=16*mm, bottomMargin=16*mm)
    styles = getSampleStyleSheet()
    story = [Paragraph("ManGo or Stay", styles["Title"]),
             Paragraph("Mango Ripeness & Surface Quality Assessment Report", styles["Heading2"]),
             Spacer(1, 5*mm)]
    rows = [
        ("Assessment ID", assessment.get("ID", "N/A")),
        ("Batch ID", assessment.get("Batch ID", "Single")),
        ("Mango filename", assessment.get("Mango", "N/A")),
        ("Date / time", assessment.get("Date").strftime("%Y-%m-%d %H:%M:%S") if assessment.get("Date") else "N/A"),
        ("Predicted ripeness", assessment.get("Ripeness", "N/A")),
        ("Model confidence", f"{assessment.get('Confidence', 0):.1f}%"),
        ("Surface quality grade", assessment.get("Grade", "N/A")),
        ("Defect percentage", f"{assessment.get('Defect %', 0):.2f}%"),
        ("Severity", assessment.get("Severity", "N/A")),
        ("Detected defect types", ", ".join(assessment.get("Defect Types", [])) or "None"),
        ("Processing information", assessment.get("Processing Info", "N/A")),
    ]
    table = Table([[Paragraph(str(k), styles["BodyText"]), Paragraph(str(v), styles["BodyText"])] for k,v in rows], colWidths=[48*mm, 116*mm])
    table.setStyle(TableStyle([("GRID", (0,0), (-1,-1), .4, colors.grey),
                               ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#E8F5E9")),
                               ("VALIGN", (0,0), (-1,-1), "TOP"), ("PADDING", (0,0), (-1,-1), 5)]))
    story += [table, Spacer(1, 6*mm)]
    for caption, image in _assessment_images(assessment):
        if image:
            story += [Paragraph(caption, styles["Heading4"]), image, Spacer(1, 4*mm)]
    doc.build(story)
    return output.getvalue()


def generate_batch_pdf(batch_id: str, assessments: list[dict]) -> bytes:
    """Generate one PDF summary with an individual page for every mango."""
    output = BytesIO()
    doc = SimpleDocTemplate(output, pagesize=A4, rightMargin=14*mm, leftMargin=14*mm,
                            topMargin=14*mm, bottomMargin=14*mm)
    styles = getSampleStyleSheet()
    story = [Paragraph("ManGo or Stay", styles["Title"]),
             Paragraph(f"Batch Assessment Report — {batch_id}", styles["Heading2"]),
             Paragraph(f"Mangoes processed: {len(assessments)}", styles["BodyText"]),
             Spacer(1, 4*mm)]
    summary = [["Filename", "Ripeness", "Confidence", "Grade", "Defect %", "Severity"]]
    for item in assessments:
        summary.append([
            item.get("Mango", ""), item.get("Ripeness", ""),
            f"{item.get('Confidence', 0):.2f}%", item.get("Grade", ""),
            f"{item.get('Defect %', 0):.2f}%", item.get("Severity", ""),
        ])
    summary_table = Table(summary, repeatRows=1, colWidths=[46*mm, 27*mm, 27*mm, 16*mm, 25*mm, 23*mm])
    summary_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), .4, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F5E9")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(summary_table)

    for item in assessments:
        story += [PageBreak(), Paragraph(str(item.get("Mango", "Mango")), styles["Heading1"])]
        details = [
            ["Assessment ID", item.get("ID", "")],
            ["Ripeness", item.get("Ripeness", "")],
            ["Confidence", f"{item.get('Confidence', 0):.2f}%"],
            ["Grade", item.get("Grade", "")],
            ["Defect percentage", f"{item.get('Defect %', 0):.2f}%"],
            ["Severity", item.get("Severity", "")],
            ["Defect types", ", ".join(item.get("Defect Types", [])) or "None"],
        ]
        detail_table = Table(details, colWidths=[45*mm, 120*mm])
        detail_table.setStyle(TableStyle([("GRID", (0,0), (-1,-1), .4, colors.grey),
                                          ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#E8F5E9"))]))
        story += [detail_table, Spacer(1, 5*mm)]
        available = [(caption, image) for caption, image in _assessment_images(
            item, width=50*mm, height=50*mm) if image]
        if available:
            story.append(Table(
                [[Paragraph(caption, styles["Heading4"]) for caption, _ in available],
                 [image for _, image in available]],
                colWidths=[56*mm] * len(available),
            ))
        else:
            story.append(Paragraph("No assessment images are available.", styles["BodyText"]))
    doc.build(story)
    return output.getvalue()
