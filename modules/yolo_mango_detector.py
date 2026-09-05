"""YOLO mango object detector."""

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mango_yolo.pt"
)


class YoloMangoDetector:
    """Detect mango objects using the trained YOLO model."""

    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        confidence_threshold=0.50,
        image_size=640,
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = float(
            confidence_threshold
        )
        self.image_size = int(image_size)

        if not self.model_path.exists():
            raise FileNotFoundError(
                "YOLO mango model was not found: "
                f"{self.model_path}"
            )

        self.model = YOLO(
            str(self.model_path)
        )

    def detect(
        self,
        image: np.ndarray,
    ) -> list:
        """
        Detect mangoes.

        Returns a list containing:
        - class id
        - class name
        - confidence
        - bounding box
        - mango crop
        """

        if image is None:
            return []

        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]

        if (
            result.boxes is None
            or len(result.boxes) == 0
        ):
            return []

        image_height, image_width = (
            image.shape[:2]
        )

        detections = []

        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )

            class_id = int(
                box.cls[0]
            )

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist(),
            )

            # Ensure coordinates remain
            # inside the image.
            x1 = max(
                0,
                min(
                    x1,
                    image_width - 1,
                ),
            )

            y1 = max(
                0,
                min(
                    y1,
                    image_height - 1,
                ),
            )

            x2 = max(
                0,
                min(
                    x2,
                    image_width,
                ),
            )

            y2 = max(
                0,
                min(
                    y2,
                    image_height,
                ),
            )

            if (
                x2 <= x1
                or y2 <= y1
            ):
                continue

            crop = image[
                y1:y2,
                x1:x2,
            ].copy()

            class_name = result.names.get(
                class_id,
                "mango",
            )

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,

                    # x1, y1, x2, y2
                    "xyxy": (
                        x1,
                        y1,
                        x2,
                        y2,
                    ),

                    # x, y, width, height
                    "bbox": (
                        x1,
                        y1,
                        x2 - x1,
                        y2 - y1,
                    ),

                    "crop": crop,
                }
            )

        return detections

    def annotate(
        self,
        image: np.ndarray,
    ):
        """
        Detect mangoes and draw bounding boxes.

        Returns:
            annotated image,
            detections
        """

        annotated = image.copy()

        detections = self.detect(
            image
        )

        for detection in detections:

            x1, y1, x2, y2 = (
                detection["xyxy"]
            )

            confidence = (
                detection["confidence"]
            )

            label = (
                f"Mango "
                f"{confidence * 100:.1f}%"
            )

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            text_y = max(
                y1 - 10,
                25,
            )

            cv2.putText(
                annotated,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if detections:

            status_text = (
                f"Mango detected: "
                f"{len(detections)}"
            )

            status_colour = (
                0,
                255,
                0,
            )

        else:

            status_text = (
                "No mango detected"
            )

            status_colour = (
                0,
                0,
                255,
            )

        cv2.putText(
            annotated,
            status_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_colour,
            2,
            cv2.LINE_AA,
        )

        return (
            annotated,
            detections,
        )