from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO


def choose_device(preference: str = "auto") -> str:
    """Choose MPS on Apple Silicon when available; otherwise CPU."""
    preference = (preference or "auto").lower()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if preference == "mps":
        if not mps_available:
            raise RuntimeError("MPS was selected but PyTorch MPS is not available. Choose auto or cpu.")
        return "mps"
    if preference == "cpu":
        return "cpu"
    return "mps" if mps_available else "cpu"


class ImageDetector:
    def __init__(self, model_path: str = "yolo11n.pt", device: str = "auto"):
        self.model_path = model_path
        self.device = choose_device(device)
        if not Path(model_path).exists() and not model_path.endswith(".pt"):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        self.model = YOLO(model_path)

    def analyze(self, image: Image.Image, confidence: float = 0.25):
        """
        Returns:
          annotated PIL image,
          DataFrame with one row per detected bounding box.
        """
        image = image.convert("RGB")
        result = self.model.predict(
            source=np.asarray(image),
            conf=float(confidence),
            device=self.device,
            verbose=False,
        )[0]

        rows = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                score = float(box.conf.item())
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                rows.append({
                    "object": str(self.model.names[class_id]),
                    "class_id": class_id,
                    "confidence": score,
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                    "box_width": round(max(0.0, x2 - x1), 2),
                    "box_height": round(max(0.0, y2 - y1), 2),
                })

        columns = [
            "object", "class_id", "confidence", "x1", "y1", "x2", "y2",
            "box_width", "box_height",
        ]
        detections = pd.DataFrame(rows, columns=columns)
        if not detections.empty:
            detections["confidence"] = detections["confidence"].round(4)

        # result.plot() uses the same confidence-filtered predictions from predict().
        annotated_array = result.plot()
        annotated = Image.fromarray(annotated_array[..., ::-1].copy())
        return annotated, detections
