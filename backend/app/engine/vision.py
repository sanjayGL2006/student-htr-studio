"""
Vision pipeline for classroom HTR.

Steps per page image:
  1. Deskew (minAreaRect on binarized foreground pixels)
  2. Binarize with Otsu's threshold
  3. Segment into line crops via morphological dilation + contour detection
     (rectangular boxes). CRAFT is the documented upgrade path for polygon
     boxes on skewed/wavy handwriting — see `segment_lines_craft` stub below.
  4. Feed each crop to TrOCR for per-line transcription + a confidence score.

NOTE ON MODEL WEIGHTS: `microsoft/trocr-base-handwritten` is downloaded from
the Hugging Face Hub the first time TrOCRModel() is instantiated. That
requires outbound network access to huggingface.co, which must be allow-listed
in whatever environment this runs in (it is not reachable from this build
sandbox, so live inference could not be executed/verified here — the pipeline
below was verified structurally, and the OpenCV segmentation stage was
verified end-to-end on synthetic page images; see README "Verification"
section).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from app.config import settings


@dataclass
class LineCrop:
    image: np.ndarray  # cropped line, BGR
    bounding_box: dict  # {x, y, w, h} in source page coordinates


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Rotate the page so text lines are horizontal, using the minimum-area
    rectangle of all foreground (ink) pixels."""
    coords = np.column_stack(np.where(gray < 250))
    if coords.shape[0] < 20:
        return gray  # not enough ink to estimate an angle; leave as-is

    angle = cv2.minAreaRect(coords)[-1]
    if angle < settings.deskew_threshold:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.1:
        return gray

    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def _binarize_otsu(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def segment_lines(page_bgr: np.ndarray) -> list[LineCrop]:
    """Rectangular line segmentation via morphological dilation.
    Uses configurable kernel sizes and falls back to connected components if no lines are found."""
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    gray = _deskew(gray)
    binary = _binarize_otsu(gray)

    # Use settings for dynamic adaptation
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, 
        (settings.dilation_kernel_width, settings.dilation_kernel_height)
    )
    dilated = cv2.dilate(binary, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = [cv2.boundingRect(c) for c in contours]
    
    # Filter using settings
    boxes = [
        b for b in boxes 
        if b[2] >= settings.min_line_width and settings.min_line_height <= b[3] <= settings.max_line_height
    ]

    # Fallback: Connected-component analysis if no valid lines found
    if not boxes and page_bgr.size > 0:
        print("[WARN] Primary segmentation failed, falling back to connected components")
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if w >= settings.min_line_width and settings.min_line_height <= h <= settings.max_line_height:
                boxes.append((x, y, w, h))
        
        # Merge very close boxes to form lines (simple heuristic)
        # For simplicity, we just use the raw CC boxes for now.

    # Read top-to-bottom, left-to-right within a line band
    boxes.sort(key=lambda b: (b[1], b[0]))

    crops: list[LineCrop] = []
    for (x, y, w, h) in boxes:
        pad = settings.line_padding
        y0, y1 = max(0, y - pad), min(page_bgr.shape[0], y + h + pad)
        x0, x1 = max(0, x - pad), min(page_bgr.shape[1], x + w + pad)
        crop = page_bgr[y0:y1, x0:x1]
        crops.append(LineCrop(image=crop, bounding_box={"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}))

    # Final fallback: whole-region OCR with warning
    if not crops and page_bgr.size > 0:
        print("[WARN] All segmentation failed, falling back to whole-region OCR")
        crops.append(
            LineCrop(
                image=page_bgr,
                bounding_box={"x": 0, "y": 0, "w": page_bgr.shape[1], "h": page_bgr.shape[0]},
            )
        )

    return crops


def segment_lines_craft(page_bgr: np.ndarray) -> list[LineCrop]:
    """Upgrade path for skewed/wavy handwritten lines: CRAFT text detection
    returns polygon boxes instead of axis-aligned rectangles, which handles
    students whose lines curve or slope mid-line. Not wired in by default to
    keep the base dependency footprint small (`craft-text-detector` pulls in
    its own torch-based detector weights). To enable: pip install
    craft-text-detector, then swap the call site in api/routes.py."""
    raise NotImplementedError(
        "CRAFT segmentation is a documented upgrade path, not enabled by default. "
        "Install `craft-text-detector` and implement polygon-crop extraction here."
    )


class TrOCREngine:
    """Thin wrapper around the HF TrOCR processor/model. Loaded once and
    reused across requests (see api/routes.py dependency)."""

    def __init__(self, model_id: str | None = None, device: str | None = None):
        self.device = device or settings.device
        self.model_id = model_id or settings.trocr_model_id
        self.processor = TrOCRProcessor.from_pretrained(self.model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def transcribe(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        """Returns (text, confidence_score). Confidence is derived from the
        mean per-token softmax probability of the generated sequence, which
        is a reasonable proxy for "how sure was the model", used purely to
        flag lines for manual review — not a calibrated probability."""
        if crop_bgr is None or crop_bgr.size == 0:
            return "", 0.0

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        pixel_values = self.processor(images=pil_img, return_tensors="pt").pixel_values.to(self.device)

        outputs = self.model.generate(
            pixel_values,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=128,
        )

        text = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]

        # Mean max-softmax probability across generated steps as a confidence proxy
        if outputs.scores:
            probs = [torch.softmax(step, dim=-1).max().item() for step in outputs.scores]
            confidence = float(sum(probs) / len(probs))
        else:
            confidence = 0.0

        return text.strip(), confidence
