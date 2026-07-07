"""
utils.py
Shared helpers: risk scoring, image annotation, image I/O, and PDF export.
Kept free of Streamlit imports so this logic can be unit tested or reused
outside the web app.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from PIL import Image

from detector import Detection

# Risk thresholds are intentionally simple and explainable - a real
# deployment would tune these against labeled historical inspection outcomes.
RISK_THRESHOLDS = {
    "LOW_MAX_DEFECTS": 1,
    "MEDIUM_MAX_DEFECTS": 4,
    "HIGH_CONFIDENCE_CUTOFF": 0.7,
}


@dataclass
class RiskAssessment:
    level: str    # "Low" | "Medium" | "High"
    score: float  # 0-100
    reason: str


def assess_risk(detections: List[Detection]) -> RiskAssessment:
    """
    Combines defect count and confidence into a simple, explainable risk
    level. More defects and higher-confidence detections both push risk up.
    This is deliberately transparent rather than a black box, so it can be
    explained line-by-line in an interview.
    """
    count = len(detections)
    if count == 0:
        return RiskAssessment("Low", 0.0, "No defects detected.")

    avg_conf = sum(d.confidence for d in detections) / count
    high_conf_count = sum(1 for d in detections if d.confidence >= RISK_THRESHOLDS["HIGH_CONFIDENCE_CUTOFF"])

    score = min(100.0, count * 12 + avg_conf * 40 + high_conf_count * 10)

    if count <= RISK_THRESHOLDS["LOW_MAX_DEFECTS"] and high_conf_count == 0:
        level = "Low"
        reason = f"{count} low-confidence defect(s) detected."
    elif count <= RISK_THRESHOLDS["MEDIUM_MAX_DEFECTS"] and high_conf_count <= 1:
        level = "Medium"
        reason = f"{count} defect(s) detected, average confidence {avg_conf:.0%}."
    else:
        level = "High"
        reason = f"{count} defect(s) detected including {high_conf_count} high-confidence finding(s)."

    return RiskAssessment(level, round(score, 1), reason)


def draw_detections(image_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """Draws bounding boxes + confidence labels onto a copy of the image."""
    annotated = image_bgr.copy()

    for det in detections:
        color = (0, 165, 255) if det.confidence < 0.7 else (0, 0, 255)  # BGR: orange / red
        cv2.rectangle(annotated, (det.x1, det.y1), (det.x2, det.y2), color, 2)
        label = f"{det.label} {det.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (det.x1, max(0, det.y1 - th - 6)),
                      (det.x1 + tw + 4, det.y1), color, -1)
        cv2.putText(annotated, label, (det.x1 + 2, max(10, det.y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Converts a PIL image (as uploaded via Streamlit) to an OpenCV BGR array."""
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(image_bgr: np.ndarray) -> Image.Image:
    """Converts an OpenCV BGR array back to a PIL image for Streamlit display."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def export_report_pdf(markdown_text: str, output_path: str) -> str:
    """
    Renders the textual report to a simple PDF using fpdf2.
    Intentionally simple formatting (strips markdown syntax rather than
    fully rendering it) - a portfolio-appropriate professional look rather
    than a full markdown-to-PDF renderer, which would be overkill here.
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", size=11)

    for line in markdown_text.splitlines():
        clean = line.replace("**", "").replace("|", "  ").replace("#", "").strip()
        if not clean:
            pdf.ln(4)
            continue
        # multi_cell(width=0) measures remaining width from the *current* x,
        # not the left margin - without resetting x, the second+ call sees
        # almost no room left and raises FPDFException. Reset before each line.
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 6, clean)

    pdf.output(output_path)
    return output_path
