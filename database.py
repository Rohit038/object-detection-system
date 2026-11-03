"""Optional PostgreSQL persistence. Configure DATABASE_URL to enable it."""
import os
from urllib.parse import urlparse

import pandas as pd


def save_analysis_if_configured(
    image_name: str,
    model_version: str,
    detections: pd.DataFrame,
    processing_time_ms: float,
) -> None:
    database_url = os.getenv("postgresql://postgres:3236@localhost:5432/object_detection")
    if not database_url:
        return

    try:
        import psycopg2
    except ImportError as exc:
        raise RuntimeError("Install psycopg2-binary to enable PostgreSQL storage.") from exc

    connection = psycopg2.connect(database_url)
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO image_analysis
                        (image_name, model_version, total_objects, processing_time_ms)
                    VALUES (%s, %s, %s, %s)
                    RETURNING image_id
                    """,
                    (image_name, model_version, int(len(detections)), float(processing_time_ms)),
                )
                image_id = cursor.fetchone()[0]

                if not detections.empty:
                    rows = []
                    for _, row in detections.iterrows():
                        rows.append((
                            image_id, str(row["object"]), int(row["class_id"]),
                            float(row["confidence"]), float(row["x1"]), float(row["y1"]),
                            float(row["x2"]), float(row["y2"]),
                            float(row["box_width"]), float(row["box_height"]),
                        ))
                    cursor.executemany(
                        """
                        INSERT INTO object_detection
                            (image_id, object_name, class_id, confidence,
                             x1, y1, x2, y2, box_width, box_height)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        rows,
                    )
    finally:
        connection.close()
