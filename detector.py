"""
detector.py
Defines the defect detection layer for VisionInspect AI.

Architecture
------------
All detectors implement the BaseDetector interface, so the rest of the
application (app.py, report.py) never needs to know which underlying
detection strategy actually produced a result. This makes it possible to
swap a hosted model for an offline fallback without touching the UI.

Detection strategy (tried in order, first available one wins):
    1. LocalYOLODetector   - offline .pt weights via Ultralytics (if configured)
    2. RoboflowDetector    - hosted pretrained crack/corrosion model via API
    3. ClassicalCVDetector - OpenCV edge/contour heuristic (always available)

This "chain of responsibility" pattern means the app degrades gracefully
instead of crashing if an API key is missing or a network call fails.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class Detection:
    """A single detected defect region."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    label: str = "defect"

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


@dataclass
class DetectionResult:
    """Full output of a detection run, including provenance."""
    detections: List[Detection] = field(default_factory=list)
    detector_name: str = "unknown"
    notes: str = ""


class BaseDetector(ABC):
    """Common interface every detection strategy must implement."""

    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this detector can run right now (keys/files present)."""

    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> DetectionResult:
        """Run detection on a BGR OpenCV image and return a DetectionResult."""


class LocalYOLODetector(BaseDetector):
    """
    Offline detector using a local Ultralytics YOLO checkpoint.

    Configure via the YOLO_WEIGHTS_PATH environment variable, pointing to a
    .pt file trained (by someone else) for crack/corrosion/defect detection.
    This is the fastest and most "production" option when available, since
    it needs no network call at inference time. Not required to run the app
    - it's a slot for a better model if you find/train one later.
    """

    name = "Local YOLO (Ultralytics)"

    def __init__(self, weights_path: Optional[str] = None, conf_threshold: float = 0.25):
        self.weights_path = weights_path or os.getenv("YOLO_WEIGHTS_PATH", "")
        self.conf_threshold = conf_threshold
        self._model = None

    def is_available(self) -> bool:
        return bool(self.weights_path) and os.path.isfile(self.weights_path)

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO  # imported lazily to keep app startup fast
            self._model = YOLO(self.weights_path)
        return self._model

    def detect(self, image_bgr: np.ndarray) -> DetectionResult:
        model = self._load()
        results = model.predict(image_bgr, conf=self.conf_threshold, verbose=False)
        detections: List[Detection] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = model.names.get(cls_id, "defect")
                detections.append(Detection(x1, y1, x2, y2, conf, label))
        return DetectionResult(detections, self.name)


class RoboflowDetector(BaseDetector):
    """
    Hosted detector using a custom Roboflow Workflow (not a bare model id).
    This targets the multi-class Pothole / Alligator Cracking / Lateral
    Cracking detection workflow provisioned in the user's own Roboflow
    workspace - a better domain fit than a generic single-class crack model,
    and validated against real pipe photos via the Roboflow Universe "Try it"
    widget before being wired in here.

    Requires ROBOFLOW_API_KEY. Workspace/workflow default to the values
    confirmed working in Roboflow's UI, but can be overridden with
    ROBOFLOW_WORKSPACE_NAME / ROBOFLOW_WORKFLOW_ID if the workflow is renamed
    or rebuilt later.

    Note on parsing: Workflow responses (especially segmentation ones) can
    nest predictions differently than a plain model `infer()` call. Rather
    than hard-coding one exact shape (which would break silently if the
    workflow is edited), `_extract_predictions` walks a few known response
    shapes. If Roboflow changes the schema in a way this doesn't handle,
    `is_available()` still returns True and the call will simply surface a
    parsing issue rather than crash the app - inspect the raw response
    (see the `debug_raw_response` flag) if detections come back empty
    unexpectedly.
    """

    name = "Roboflow Hosted Workflow"
    DEFAULT_WORKSPACE_NAME = "shahwar"
    DEFAULT_WORKFLOW_ID = "general-segmentation-api"
    DEFAULT_CLASSES = ["Pothole", "Alligator Cracking", "Lateral Cracking"]

    def __init__(self, api_key: Optional[str] = None, workspace_name: Optional[str] = None,
                 workflow_id: Optional[str] = None, classes: Optional[List[str]] = None,
                 conf_threshold: float = 0.3, debug_raw_response: bool = False):
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY", "")
        self.workspace_name = workspace_name or os.getenv("ROBOFLOW_WORKSPACE_NAME", self.DEFAULT_WORKSPACE_NAME)
        self.workflow_id = workflow_id or os.getenv("ROBOFLOW_WORKFLOW_ID", self.DEFAULT_WORKFLOW_ID)
        env_classes = os.getenv("ROBOFLOW_CLASSES", "")
        self.classes = classes or ([c.strip() for c in env_classes.split(",") if c.strip()] or self.DEFAULT_CLASSES)
        self.conf_threshold = conf_threshold
        self.debug_raw_response = debug_raw_response
        self._client = None

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            from inference_sdk import InferenceHTTPClient
            self._client = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key=self.api_key,
            )
        return self._client

    def detect(self, image_bgr: np.ndarray) -> DetectionResult:
        client = self._get_client()
        # This workflow has a required `classes` input (the same list shown
        # as selectable buttons in Roboflow's own "Try it" widget) that feeds
        # a SAM segmentation step - omitting it causes a 400 error, so it's
        # always passed explicitly rather than relying on a server-side default.
        raw = client.run_workflow(
            workspace_name=self.workspace_name,
            workflow_id=self.workflow_id,
            images={"image": image_bgr},
            parameters={"classes": self.classes},
        )

        if self.debug_raw_response:
            print("RAW ROBOFLOW WORKFLOW RESPONSE:", raw)

        predictions = self._extract_predictions(raw)

        detections: List[Detection] = []
        for pred in predictions:
            conf = float(pred.get("confidence", 0))
            if conf < self.conf_threshold:
                continue

            if pred.get("points"):
                # Segmentation polygon - derive a bounding box from the point cloud
                xs = [p["x"] for p in pred["points"]]
                ys = [p["y"] for p in pred["points"]]
                x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            elif all(k in pred for k in ("x", "y", "width", "height")):
                cx, cy, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                x1, y1 = int(cx - w / 2), int(cy - h / 2)
                x2, y2 = int(cx + w / 2), int(cy + h / 2)
            else:
                continue  # unrecognized prediction shape - skip rather than guess wrong

            label = pred.get("class", pred.get("class_name", "defect"))
            detections.append(Detection(x1, y1, x2, y2, round(conf, 2), label))

        return DetectionResult(detections, self.name)

    @staticmethod
    def _extract_predictions(raw) -> List[dict]:
        """
        Workflow responses vary based on how the workflow's output blocks
        are configured. This walks the common shapes (a list wrapping one
        result per input image, each holding named output blocks) and pulls
        out whatever list of per-object predictions it can find, so a minor
        workflow edit in the Roboflow UI doesn't silently break detection.
        """
        if isinstance(raw, list) and raw:
            raw = raw[0]

        if isinstance(raw, dict):
            for value in raw.values():
                if isinstance(value, dict) and "predictions" in value:
                    inner = value["predictions"]
                    if isinstance(inner, dict) and "predictions" in inner:
                        return inner["predictions"]
                    if isinstance(inner, list):
                        return inner
                if isinstance(value, list) and value and isinstance(value[0], dict) and "confidence" in value[0]:
                    return value

        return []


class ClassicalCVDetector(BaseDetector):
    """
    Dependency-free fallback detector using classical image processing:
    grayscale -> blur -> Canny edges -> contour filtering.

    This never fails and never needs network access, which makes it the
    guaranteed final link in the detection chain. It flags elongated,
    high-edge-density regions as "possible defect" candidates - a reasonable
    proxy for cracks in the absence of a trained model, and fully explainable
    to a non-ML audience.
    """

    name = "Classical CV Heuristic"

    def __init__(self, min_area: int = 150, max_candidates: int = 25):
        self.min_area = min_area
        self.max_candidates = max_candidates

    def is_available(self) -> bool:
        return True  # always available - the safety net of the chain

    def detect(self, image_bgr: np.ndarray) -> DetectionResult:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = max(w, h) / max(1, min(w, h))
            edge_density = area / (w * h + 1e-6)
            # Elongated, irregular (low fill-ratio) shapes score as more crack-like
            confidence = min(0.95, 0.3 + 0.1 * min(aspect_ratio, 5) + 0.2 * (1 - edge_density))
            candidates.append((confidence, Detection(x, y, x + w, y + h, round(confidence, 2), "possible_defect")))

        candidates.sort(key=lambda t: t[0], reverse=True)
        detections = [d for _, d in candidates[: self.max_candidates]]

        return DetectionResult(
            detections,
            self.name,
            notes=("Heuristic detector: flags edge-dense, elongated regions as candidates. "
                   "Treat results as areas for human review, not certainties.")
        )


def get_detector_chain() -> List[BaseDetector]:
    """Detectors in priority order: local YOLO -> Roboflow -> classical CV."""
    return [LocalYOLODetector(), RoboflowDetector(), ClassicalCVDetector()]


def run_detection(image_bgr: np.ndarray) -> DetectionResult:
    """
    Tries each detector in priority order and returns the first available
    one's result. ClassicalCVDetector is always available, so in practice
    this never raises - it just degrades to the simplest strategy.

    Any earlier failures are recorded into the successful result's `notes`
    field, so they're visible directly in the app UI (via the annotated
    image caption) rather than only in server logs - which is more reliable
    to actually see, especially on hosted deployments.
    """
    errors = []
    for detector in get_detector_chain():
        if not detector.is_available():
            continue
        try:
            result = detector.detect(image_bgr)
            if errors:
                prior = " | ".join(errors)
                failure_note = f"[Note: earlier detector(s) failed first - {prior}]"
                result.notes = f"{result.notes} {failure_note}".strip() if result.notes else failure_note
            return result
        except Exception as exc:  # bad key, network hiccup, etc. - fall through
            errors.append(f"{detector.name}: {type(exc).__name__}: {exc}")
            continue
    raise RuntimeError(f"All detectors failed. Errors: {' | '.join(errors)}")
