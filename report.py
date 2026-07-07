"""
report.py
Turns raw detections + a risk assessment into a human-readable inspection
report. Two interchangeable strategies implement the same interface, so
app.py never needs to know or care which one actually ran:

    RuleBasedReportGenerator - deterministic, offline, always works.
    ClaudeReportGenerator    - wraps the same structured data and asks
                               Claude (Anthropic API) to write a more
                               natural, engineer-facing narrative.
                               Optional - silently falls back to the
                               rule-based summary if no key is set or
                               the call fails for any reason.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

from detector import Detection
from utils import RiskAssessment


@dataclass
class InspectionReport:
    image_name: str
    generated_at: str
    detector_used: str
    defect_count: int
    risk_level: str
    risk_score: float
    summary: str
    defects: List[Detection] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "# Inspection Report",
            "",
            f"**Image:** {self.image_name}  ",
            f"**Generated:** {self.generated_at}  ",
            f"**Detection method:** {self.detector_used}  ",
            f"**Defects detected:** {self.defect_count}  ",
            f"**Risk level:** {self.risk_level} (score: {self.risk_score}/100)  ",
            "",
            "## Summary",
            self.summary,
            "",
            "## Detected Defects",
            "",
            "| # | Label | Confidence | Location (x1, y1, x2, y2) |",
            "|---|-------|------------|----------------------------|",
        ]
        for i, d in enumerate(self.defects, start=1):
            lines.append(f"| {i} | {d.label} | {d.confidence:.0%} | ({d.x1}, {d.y1}, {d.x2}, {d.y2}) |")
        if not self.defects:
            lines.append("| - | No defects detected | - | - |")
        return "\n".join(lines)


class BaseReportGenerator(ABC):
    @abstractmethod
    def generate(self, image_name: str, detector_used: str,
                 detections: List[Detection], risk: RiskAssessment) -> InspectionReport:
        ...


class RuleBasedReportGenerator(BaseReportGenerator):
    """Deterministic, templated report. No external dependencies, never fails."""

    def generate(self, image_name, detector_used, detections, risk) -> InspectionReport:
        if not detections:
            summary = (
                "No defects were detected in this image. The inspected surface "
                "appears to be in acceptable condition based on the current model. "
                "Routine re-inspection is recommended per standard schedule."
            )
        else:
            labels = sorted({d.label for d in detections})
            follow_up = {
                "High": "Immediate follow-up inspection is recommended.",
                "Medium": "A follow-up inspection is advised within the standard maintenance window.",
                "Low": "No immediate action is required beyond routine monitoring.",
            }[risk.level]
            summary = (
                f"The inspection identified {len(detections)} potential defect(s) "
                f"({', '.join(labels)}) with an overall risk level of {risk.level}. "
                f"{risk.reason} {follow_up}"
            )

        return InspectionReport(
            image_name=image_name,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            detector_used=detector_used,
            defect_count=len(detections),
            risk_level=risk.level,
            risk_score=risk.score,
            summary=summary,
            defects=detections,
        )


class ClaudeReportGenerator(BaseReportGenerator):
    """
    Optional enhancement: computes the same structured data as
    RuleBasedReportGenerator, then asks Claude to rewrite the summary as a
    natural, engineer-facing narrative. Uses Haiku since this is a short,
    low-latency text task - no need for a larger model here.
    """

    MODEL = "claude-haiku-4-5-20251001"

    def __init__(self):
        self._base = RuleBasedReportGenerator()
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, image_name, detector_used, detections, risk) -> InspectionReport:
        base_report = self._base.generate(image_name, detector_used, detections, risk)

        if not self.is_available():
            return base_report

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)

            defect_lines = "\n".join(
                f"- {d.label}, confidence {d.confidence:.0%}, box ({d.x1},{d.y1},{d.x2},{d.y2})"
                for d in detections
            ) or "- none detected"

            prompt = (
                "You are an industrial inspection assistant writing the summary section "
                "of a defect inspection report for a pipeline/infrastructure operator. "
                "Write 3-5 sentences, professional tone, no bullet points, no headers. "
                f"Risk level: {risk.level} (score {risk.score}/100). Reason: {risk.reason}\n"
                f"Detected defects:\n{defect_lines}\n"
                "Do not invent defects beyond what is listed above."
            )

            response = client.messages.create(
                model=self.MODEL,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            narrative = "".join(
                block.text for block in response.content if block.type == "text"
            ).strip()

            if narrative:
                base_report.summary = narrative

        except Exception:
            # Any failure (bad key, network, rate limit) - keep the rule-based summary.
            # A production system would log this; here we fail silently by design
            # so a missing/invalid key never breaks the demo.
            pass

        return base_report


def get_report_generator(use_llm: bool = False) -> BaseReportGenerator:
    """Factory: returns the Claude-enhanced generator only if requested AND available."""
    if use_llm:
        llm_gen = ClaudeReportGenerator()
        if llm_gen.is_available():
            return llm_gen
    return RuleBasedReportGenerator()
