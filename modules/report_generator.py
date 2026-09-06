"""Detailed assessment reports. PDF dependencies are loaded only on export."""
import base64
import json
import math
from datetime import datetime
from io import BytesIO
from xml.sax.saxutils import escape

import cv2
import numpy as np

GRADING_RULES = [
    ('Ripe', '<=5%', '<=5%', 'Premium', 'Best maturity condition and minimal surface defects'),
    ('Semi-Ripe', '<=5%', '<=5%', 'Grade 1', 'Good quality but not yet fully ripe'),
    ('Unripe', '<=5%', '<=5%', 'Grade 2', 'Clean surface but immature'),
    ('Ripe', '>5% and <=10%', '<=5%', 'Grade 1', 'Good maturity with moderate cosmetic blemish'),
    ('Semi-Ripe', '>5% and <=10%', '<=5%', 'Grade 2', 'Incomplete maturity plus noticeable blemish'),
    ('Unripe', '>5% and <=10%', '<=5%', 'Grade 2', 'Already limited by immature condition'),
    ('Ripe', '<=10%', '>5% and <=10%', 'Grade 2', 'Correct maturity but noticeable physical damage'),
    ('Semi-Ripe', '<=10%', '>5% and <=10%', 'Grade 2', 'Incomplete maturity and damage'),
    ('Unripe', '<=10%', '>5% and <=10%', 'Grade 2', 'Immature and damaged'),
    ('Any', '>10%', 'Any', 'Reject', 'Excessive blemish'),
    ('Any', 'Any', '>10%', 'Reject', 'Excessive damage'),
    ('Rotten', 'Any', 'Any', 'Reject', 'Deteriorated fruit'),
]


def format_coverage(value):
    try:
        number = float(value)
        return f'{number:.2f}%' if math.isfinite(number) else 'Unavailable'
    except (TypeError, ValueError):
        return 'Unavailable'


def build_report_details(analysis):
    """Collect actual intermediate outputs, never recompute historical results."""
    source = analysis.get('result', {})
    defects = analysis.get('blemish_result', {})
    images = {k: source[k] for k in (
        'resized', 'gaussian', 'hue', 'saturation', 'value', 'candidate_mask',
        'leaf_mask', 'fruit_mask', 'segmented_bgr', 'model_segmented_bgr',
    ) if source.get(k) is not None}
    if source.get('enhanced') is not None:
        images['enhanced_bgr'] = cv2.cvtColor(source['enhanced'], cv2.COLOR_HSV2BGR)
    for key in ('blemish_mask', 'damage_only_mask', 'damage_mask', 'blackhat', 'overlay'):
        if defects.get(key) is not None:
            images[key] = defects[key]
    details = {k: analysis.get(k) for k in (
        'probabilities', 'hsv_features', 'statistical_analysis', 'preprocessing_time',
        'inference_time_ms', 'feature_count', 'mango_pixel_count', 'total_pixel_count',
        'accepted_count', 'classification_error',
    )}
    details.update(version=1, images=images, original_size=source.get('original_size'),
                   blemish_pixel_area=defects.get('blemish_pixel_area'),
                   damage_pixel_area=defects.get('damage_pixel_area'))
    details['objects'] = [dict(object=o['object_index'], ripeness=r['prediction'],
                               confidence=r['confidence'])
                          for o, r in analysis.get('object_predictions', [])]
    return details


def serialize_report_details(analysis):
    """Persist report-sized PNGs while keeping numerical measurements native."""
    details = build_report_details(analysis)
    packed = {}
    for key, image in details['images'].items():
        h, w = image.shape[:2]
        scale = min(1.0, 900 / max(h, w))
        if scale < 1:
            image = cv2.resize(image, (max(1, round(w*scale)), max(1, round(h*scale))),
                               interpolation=cv2.INTER_NEAREST if image.ndim == 2 else cv2.INTER_AREA)
        ok, buffer = cv2.imencode('.png', image)
        if not ok:
            raise ValueError(f'Could not store report image: {key}')
        packed[key] = base64.b64encode(buffer).decode('ascii')
    details['images'] = packed
    return json.dumps(details, default=lambda v: v.item() if isinstance(v, np.generic) else str(v))


def deserialize_report_details(value):
    if not value:
        return {}
    details = json.loads(value)
    details['images'] = {k: cv2.imdecode(np.frombuffer(base64.b64decode(v), np.uint8),
                                         cv2.IMREAD_UNCHANGED)
                         for k, v in details.get('images', {}).items()}
    return details


def generate_report_string(fruit_type='Harumanis mango', batch_id='', ripeness=None,
                           confidence=None, quality_result=None, severity=None,
                           defect_types=None, report_details=None):
    q = quality_result or {}
    d = report_details or {}
    lines = ['# Mango Assessment Report', '', f'Fruit: {fruit_type}', f'Batch: {batch_id or "Unavailable"}',
             f'Ripeness: {ripeness or "Unavailable"}', f'Confidence: {format_coverage(confidence)}',
             f'Blemish coverage: {format_coverage(q.get("blemish_coverage"))}',
             f'Damage coverage: {format_coverage(q.get("damage_coverage"))}',
             f'Overall grade: {q.get("grade", "Unavailable")}',
             f'Reason: {q.get("reason", "Unavailable")}',
             f'Severity: {severity or "Unavailable"}',
             f'Detected defects: {", ".join(defect_types or []) or "Unavailable"}', '',
             '## Preprocessing', 'Background estimation, GrabCut body extraction, thin-attachment removal and model resize.',
             f'Fruit pixels: {d.get("mango_pixel_count", "Unavailable")}', '', '## Ripeness probabilities']
    lines.extend(f'- {k}: {format_coverage(v)}' for k, v in (d.get('probabilities') or {}).items())
    lines += ['', '## Grading reference', '| Ripeness | Blemish | Damage | Grade | Interpretation |',
              '|---|---|---|---|---|']
    lines.extend('| ' + ' | '.join(row) + ' |' for row in GRADING_RULES)
    lines += ['', 'Reference: user-provided rule based grading table. Reject conditions take precedence.']
    return '\n'.join(lines)


def generate_pdf_report(fruit_type='Harumanis mango', batch_id='', ripeness=None,
                        confidence=None, quality_result=None, severity=None,
                        defect_types=None, original_image=None, overlay_image=None,
                        report_details=None):
    """Return PDF bytes. Colour images use OpenCV BGR; masks are grayscale."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    except ImportError as exc:
        raise RuntimeError('PDF export requires ReportLab: python -m pip install reportlab') from exc
    q, d = quality_result or {}, report_details or {}
    pictures = d.get('images', {})
    green, ink, pale, line = [colors.HexColor(c) for c in ('#315E4D', '#25342F', '#EEF4F0', '#D4DED8')]
    styles = getSampleStyleSheet()
    for name, size, leading, colour, bold in (
        ('TitleReport', 23, 28, green, True), ('Section', 15, 19, green, True),
        ('Cell', 9, 13, ink, False), ('SmallCell', 8, 11, ink, False), ('WhiteCell', 8, 11, colors.white, True),
    ):
        styles.add(ParagraphStyle(name=name, fontName='Helvetica-Bold' if bold else 'Helvetica',
                                  fontSize=size, leading=leading, textColor=colour, spaceAfter=0))
    width = A4[0] - 80
    story = []
    def p(text, style='Cell'):
        return Paragraph(escape(str(text if text is not None else 'Unavailable')).replace('\n', '<br/>'), styles[style])
    def metric(v, suffix=''):
        if v is None:
            return 'Unavailable'
        return (f'{v:,.2f}' if isinstance(v, (float, np.floating)) else str(v)) + suffix
    def table(rows, widths, header=False):
        cells = [[p(v, 'WhiteCell' if header and i == 0 else 'SmallCell') for v in row] for i, row in enumerate(rows)]
        t = Table(cells, colWidths=widths, repeatRows=int(header), hAlign='LEFT')
        commands = [('VALIGN', (0,0), (-1,-1), 'TOP'), ('GRID', (0,0), (-1,-1), .45, line),
                    ('LEFTPADDING', (0,0), (-1,-1), 8), ('RIGHTPADDING', (0,0), (-1,-1), 8),
                    ('TOPPADDING', (0,0), (-1,-1), 7), ('BOTTOMPADDING', (0,0), (-1,-1), 7)]
        commands += [('BACKGROUND', (0,0), (-1,0), green), ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, pale])] if header else [('BACKGROUND', (0,0), (0,-1), pale)]
        t.setStyle(TableStyle(commands))
        return t
    def photo(array, box_w, box_h):
        if array is None:
            t = Table([[p('Not recorded for this assessment')]], colWidths=[box_w], rowHeights=[box_h])
            t.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('BACKGROUND', (0,0), (-1,-1), pale)]))
            return t
        ok, buffer = cv2.imencode('.png', np.asarray(array))
        if not ok:
            raise ValueError('Could not encode a report image.')
        h, w = array.shape[:2]
        factor = min(box_w/w, box_h/h)
        return Image(BytesIO(buffer.tobytes()), width=w*factor, height=h*factor, hAlign='CENTER')
    def grid(items, box_h=145):
        for index in range(0, len(items), 2):
            cells = [[photo(a, width/2-22, box_h), Spacer(1,5), p(label, 'SmallCell')]
                     for label, a in items[index:index+2]]
            if len(cells) == 1:
                cells.append('')
            t = Table([cells], colWidths=[width/2]*2)
            t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'), ('GRID',(0,0),(-1,-1),.45,line),
                                   ('TOPPADDING',(0,0),(-1,-1),10), ('BOTTOMPADDING',(0,0),(-1,-1),10)]))
            story.extend([t, Spacer(1,10)])
    def section(title):
        story.extend([p(title, 'Section'), Spacer(1,12)])
    story += [p('Mango Assessment Report', 'TitleReport'), Spacer(1,10), p(f'{fruit_type} | Batch: {batch_id or "Unavailable"}'),
              p('Generated '+datetime.now().strftime('%d %b %Y, %H:%M'), 'SmallCell'), Spacer(1,20)]
    summary = [['Result','Assessment'], ['Ripeness',ripeness or 'Unavailable'], ['Confidence',format_coverage(confidence)],
               ['Blemish coverage',format_coverage(q.get('blemish_coverage'))], ['Damage coverage',format_coverage(q.get('damage_coverage'))],
               ['Overall grade',q.get('grade','Unavailable')], ['Severity',severity or 'Unavailable'], ['Mangoes detected',metric(d.get('accepted_count'))]]
    layout = Table([[[photo(original_image,205,260), Spacer(1,6), p('User input image','SmallCell')],
                     [table(summary,[100,width-333],True)]]], colWidths=[223,width-223])
    layout.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'), ('LEFTPADDING',(0,0),(-1,-1),0), ('RIGHTPADDING',(0,0),(-1,-1),10)]))
    story += [layout, Spacer(1,20)]
    section('Overall interpretation')
    story += [p(q.get('reason','Grading information was not recorded.')), Spacer(1,10), p('Detected types: '+(', '.join(defect_types or []) or 'Unavailable'))]
    if d.get('classification_error'):
        story += [Spacer(1,10),p(d['classification_error'])]
    if (d.get('accepted_count') or 0) > 1:
        story += [Spacer(1,10), p('Ripeness summary refers to the first mango. Surface measurements combine accepted mangoes; individual predictions follow.')]
    story += [PageBreak()]
    section('1. Preprocessing - image preparation')
    story += [p('Original colours are preserved for analysis. HSV channels and contrast enhancement are diagnostic views, not a hard foreground cutoff.'), Spacer(1,10)]
    grid([(label,pictures.get(key)) for label,key in (
        ('1. Resized / letterboxed image','resized'), ('2. Gaussian-filtered image','gaussian'),
        ('3. Hue channel (grayscale view)','hue'), ('4. Saturation channel','saturation'),
        ('5. Value / brightness channel','value'), ('6. Enhanced diagnostic image','enhanced_bgr'))],125)
    story += [PageBreak()]
    section('2. Preprocessing - fruit body extraction')
    story += [p('Background estimation supplies candidates. GrabCut refines the body; morphology removes thin attachments and seals narrow cuts. Overlapping leaves may remain.'), Spacer(1,10)]
    grid([(label,pictures.get(key)) for label,key in (
        ('7. Foreground candidate','candidate_mask'), ('8. Removed thin attachments','leaf_mask'),
        ('9. Final fruit-body mask','fruit_mask'), ('10. Fruit body used for analysis','segmented_bgr'))],160)
    story += [table([['Preprocessing time',metric(d.get('preprocessing_time'),' s')],
                     ['Fruit pixels / image pixels',f'{metric(d.get("mango_pixel_count"))} / {metric(d.get("total_pixel_count"))}'],
                     ['Original array dimensions',metric(d.get('original_size'))]],[170,width-170]), PageBreak()]
    section('3. Ripeness classification')
    story += [table([['Predicted stage',ripeness or 'Unavailable'], ['Confidence',format_coverage(confidence)],
                     ['Inference time',metric(d.get('inference_time_ms'),' ms')], ['Feature count',metric(d.get('feature_count'))]],[170,width-170]), Spacer(1,14)]
    probabilities = d.get('probabilities') or {}
    story += [table([['Ripeness class','Probability']]+[[str(k).replace('_',' ').title(),format_coverage(v)] for k,v in probabilities.items()], [width*.6,width*.4], True) if probabilities else p('Class probabilities were not recorded.'), Spacer(1,16)]
    section('Colour and statistical measurements')
    measurements = {**(d.get('hsv_features') or {}), **(d.get('statistical_analysis') or {})}
    story += [table([['Measurement','Value']]+[[str(k).replace('_',' ').title(),metric(v)] for k,v in measurements.items()], [width*.6,width*.4], True) if measurements else p('Detailed measurements were not recorded.')]
    if d.get('objects'):
        story += [Spacer(1,16), p('Individual mango predictions','Section'), Spacer(1,10),
                  table([['Object','Ripeness','Confidence']]+[[o['object'],o['ripeness'],format_coverage(o['confidence'])] for o in d['objects']], [70,width-170,100],True)]
    story += [PageBreak()]
    section('4. Blemish and damage analysis')
    story += [table([['Measure','Blemish','Damage'], ['Coverage',format_coverage(q.get('blemish_coverage')),format_coverage(q.get('damage_coverage'))],
                     ['Pixel area',metric(d.get('blemish_pixel_area')),metric(d.get('damage_pixel_area'))]], [width*.4,width*.3,width*.3],True), Spacer(1,10)]
    grid([('Blemish mask',pictures.get('blemish_mask')), ('Damage-only mask',pictures.get('damage_only_mask')),
          ('Combined defect mask',pictures.get('damage_mask')), ('Defect overlay',overlay_image if overlay_image is not None else pictures.get('overlay'))],165)
    story += [p('Detected types: '+(', '.join(defect_types or []) or 'Unavailable')), Spacer(1,6),
              p('Coverage uses the stored detector measurements. Combined defect coverage can differ from the sum when masks overlap.'), PageBreak()]
    section('5. Overall grade and rule reference')
    story += [table([['Final grade',q.get('grade','Unavailable')], ['Reason',q.get('reason','Unavailable')],
                     ['Ripeness',ripeness or 'Unavailable'], ['Blemish coverage',format_coverage(q.get('blemish_coverage'))],
                     ['Damage coverage',format_coverage(q.get('damage_coverage'))]],[140,width-140]), Spacer(1,18)]
    story += [table([['Ripeness','Blemish coverage','Damage coverage','Final grade','Interpretation']]+GRADING_RULES,
                    [65,86,86,64,width-301],True), Spacer(1,12),
              p('Reference: user-provided rule based grading.png. Reject rules take precedence: Rotten, blemish above 10%, or damage above 10%. The 5% and 10% limits are inclusive where indicated.','SmallCell')]
    output = BytesIO()
    doc = SimpleDocTemplate(output,pagesize=A4,rightMargin=40,leftMargin=40,topMargin=42,bottomMargin=42,
                            title='Mango Assessment Report',author='ManGo or Stay')
    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(line); canvas.line(40,30,A4[0]-40,30)
        canvas.setFont('Helvetica',8); canvas.setFillColor(ink)
        canvas.drawString(40,18,'ManGo or Stay | Mango assessment')
        canvas.drawRightString(A4[0]-40,18,f'Page {document.page}')
        canvas.restoreState()
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    return output.getvalue()
