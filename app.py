"""
app.py
Streamlit UI for VisionInspect AI.

This file is intentionally thin: it wires together the detection layer
(detector.py), the report layer (report.py), and shared helpers (utils.py).
No detection or report logic lives here - that separation keeps the app
testable and lets each layer be swapped independently.
"""

import os
import tempfile
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from detector import run_detection, get_detector_chain
from report import get_report_generator
from utils import assess_risk, draw_detections, pil_to_bgr, bgr_to_pil, export_report_pdf

load_dotenv()

st.set_page_config(page_title="VisionInspect AI", page_icon="\U0001F50D", layout="wide")


def render_header():
    st.title("\U0001F50D VisionInspect AI")
    st.caption(
        "Automated first-pass visual inspection for industrial infrastructure - "
        "pipelines, tunnels, bridges, and equipment. Upload an inspection image "
        "to detect defects, assess risk, and generate a report."
    )
    with st.expander("How this works"):
        st.markdown(
            "- **Detection** tries a chain of strategies in order: a local YOLO model "
            "(if configured), a hosted Roboflow crack-detection model, and finally a "
            "dependency-free classical computer vision heuristic. The first available "
            "strategy runs, and the app shows which one was used.\n"
            "- **Risk scoring** combines defect count and detection confidence into a "
            "Low / Medium / High rating.\n"
            "- **Report generation** is rule-based by default, with an optional "
            "Claude-powered narrative summary."
        )


def render_sidebar() -> bool:
    st.sidebar.header("Settings")
    use_llm = st.sidebar.toggle(
        "Enhance summary with Claude",
        value=False,
        help="Requires ANTHROPIC_API_KEY. Falls back to the rule-based summary if unavailable.",
    )
    st.sidebar.divider()
    st.sidebar.subheader("Detector availability")
    for d in get_detector_chain():
        status = "\u2705 ready" if d.is_available() else "\u26D4 not configured"
        st.sidebar.write(f"**{d.name}**: {status}")
    return use_llm


def main():
    render_header()
    use_llm = render_sidebar()

    uploaded_file = st.file_uploader("Upload an inspection image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("Upload a JPG or PNG image to begin.")
        return

    image = Image.open(uploaded_file)
    image_bgr = pil_to_bgr(image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)

    with st.spinner("Running defect detection..."):
        result = run_detection(image_bgr)

    annotated_bgr = draw_detections(image_bgr, result.detections)
    annotated_image = bgr_to_pil(annotated_bgr)

    with col2:
        st.subheader(f"Annotated ({result.detector_name})")
        st.image(annotated_image, use_container_width=True)
        if result.notes:
            st.caption(result.notes)

    risk = assess_risk(result.detections)

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Defects detected", len(result.detections))
    m2.metric("Risk level", risk.level)
    m3.metric("Risk score", f"{risk.score}/100")

    if result.detections:
        st.subheader("Detected defects")
        table_data = [
            {
                "#": i + 1,
                "Label": d.label,
                "Confidence": f"{d.confidence:.0%}",
                "Location": f"({d.x1}, {d.y1}, {d.x2}, {d.y2})",
            }
            for i, d in enumerate(result.detections)
        ]
        st.dataframe(table_data, use_container_width=True, hide_index=True)
    else:
        st.success("No defects detected.")

    st.divider()
    st.subheader("Inspection report")

    generator = get_report_generator(use_llm=use_llm)
    report = generator.generate(
        image_name=uploaded_file.name,
        detector_used=result.detector_name,
        detections=result.detections,
        risk=risk,
    )
    st.markdown(report.summary)

    report_md = report.to_markdown()
    dl1, dl2 = st.columns(2)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with dl1:
        st.download_button(
            "Download report (Markdown)",
            data=report_md,
            file_name=f"inspection_report_{timestamp}.md",
            mime="text/markdown",
        )
    with dl2:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            export_report_pdf(report_md, tmp_path)
            with open(tmp_path, "rb") as f:
                st.download_button(
                    "Download report (PDF)",
                    data=f.read(),
                    file_name=f"inspection_report_{timestamp}.pdf",
                    mime="application/pdf",
                )
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    main()
