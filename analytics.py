import pandas as pd


def summarize_detections(detections: pd.DataFrame) -> pd.DataFrame:
    columns = ["object", "count", "avg_confidence", "min_confidence", "max_confidence"]
    if detections is None or detections.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        detections.groupby("object", dropna=False)
        .agg(
            count=("object", "size"),
            avg_confidence=("confidence", "mean"),
            min_confidence=("confidence", "min"),
            max_confidence=("confidence", "max"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )
    for col in ["avg_confidence", "min_confidence", "max_confidence"]:
        summary[col] = summary[col].round(3)
    return summary[columns]


def build_kpis(detections: pd.DataFrame) -> dict:
    if detections is None or detections.empty:
        return {
            "total_objects": 0,
            "unique_categories": 0,
            "average_confidence": 0.0,
        }
    return {
        "total_objects": int(len(detections)),
        "unique_categories": int(detections["object"].nunique()),
        "average_confidence": float(detections["confidence"].mean()),
    }


def combine_image_summaries(detections: pd.DataFrame) -> pd.DataFrame:
    if detections is None or detections.empty:
        return pd.DataFrame(columns=["image_name", "object", "count"])
    return (
        detections.groupby(["image_name", "object"])
        .size()
        .reset_index(name="count")
        .sort_values(["image_name", "count"], ascending=[True, False])
    )


def counting_mae(actual_counts, predicted_counts) -> float:
    """Mean absolute error for paired actual and predicted counts."""
    if len(actual_counts) != len(predicted_counts):
        raise ValueError("actual_counts and predicted_counts must have equal lengths")
    if len(actual_counts) == 0:
        raise ValueError("At least one count pair is required")
    errors = [abs(float(a) - float(p)) for a, p in zip(actual_counts, predicted_counts)]
    return sum(errors) / len(errors)
