import os
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from detector import ImageDetector, choose_device
from analytics import summarize_detections, build_kpis, combine_image_summaries
from database import save_analysis_if_configured

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="AI Image Intelligence",
    page_icon="🔎",
    layout="wide",
)

st.title("AI Image Intelligence")
st.caption("Computer Vision • Machine Learning • Business Analytics")

with st.sidebar:
    st.header("Model settings")
    default_model = os.getenv("AI_IMAGE_MODEL", "yolo11n.pt")
    model_path = st.text_input(
        "Model checkpoint",
        value=default_model,
        help="Use yolo11n.pt for pretrained baseline or a trained best.pt path.",
    )
    confidence = st.slider(
        "Minimum confidence", min_value=0.05, max_value=0.95,
        value=0.25, step=0.05,
    )
    device = st.selectbox(
        "Compute device",
        options=["auto", "mps", "cpu"],
        index=0,
        help="MPS is Apple's Metal acceleration. If it errors, choose CPU.",
    )
    st.divider()
    st.caption("The baseline model only detects classes it has learned.")
    st.caption("Gender is not inferred from appearance. People counts are reported separately.")

@st.cache_resource(show_spinner="Loading object detection model...")
def load_detector(checkpoint: str, selected_device: str):
    return ImageDetector(checkpoint, selected_device)

try:
    detector = load_detector(model_path.strip(), device)
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.info("Check the checkpoint path and internet connection for the initial pretrained-weight download.")
    st.stop()

uploads = st.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if not uploads:
    st.info("Upload images to see detections, object counts, charts, and a CSV report.")
    st.markdown(
        """
        **Recommended test set:** a desk/stationery photo, an animal photo, a single-person photo,
        and a group photo. Record missed objects and incorrect labels before training a custom model.
        """
    )
    st.stop()

if st.button("Analyze uploaded images", type="primary", use_container_width=True):
    all_detection_frames = []
    image_rows = []
    progress = st.progress(0, text="Starting analysis...")

    for index, uploaded in enumerate(uploads):
        try:
            image = Image.open(uploaded).convert("RGB")
            start = time.perf_counter()
            annotated, detections = detector.analyze(
                image=image,
                confidence=confidence,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            image_id = f"{index + 1:03d}"
            detections.insert(0, "image_name", uploaded.name)
            detections.insert(0, "image_id", image_id)
            all_detection_frames.append(detections)

            image_rows.append({
                "image_id": image_id,
                "image_name": uploaded.name,
                "total_objects": len(detections),
                "unique_categories": int(detections["object"].nunique()) if not detections.empty else 0,
                "processing_time_ms": round(elapsed_ms, 1),
                "model_version": Path(model_path).name,
            })

            with st.expander(f"{uploaded.name} — {len(detections)} detected object(s)", expanded=(index == 0)):
                left, right = st.columns([1.2, 1])
                with left:
                    st.image(annotated, caption="Detected objects", use_container_width=True)
                with right:
                    kpis = build_kpis(detections)
                    a, b = st.columns(2)
                    a.metric("Detected objects", kpis["total_objects"])
                    b.metric("Categories", kpis["unique_categories"])
                    st.metric("Mean detection confidence", f'{kpis["average_confidence"]:.1%}')
                    st.dataframe(
                        summarize_detections(detections),
                        use_container_width=True,
                        hide_index=True,
                    )

            # Optional database persistence; no connection is attempted unless DATABASE_URL is set.
            save_analysis_if_configured(
                image_name=uploaded.name,
                model_version=Path(model_path).name,
                detections=detections,
                processing_time_ms=elapsed_ms,
            )
        except Exception as exc:
            st.error(f"Could not analyze {uploaded.name}: {exc}")

        progress.progress((index + 1) / len(uploads), text=f"Processed {index + 1}/{len(uploads)} images")

    if all_detection_frames:
        combined = pd.concat(all_detection_frames, ignore_index=True)
        image_summary = pd.DataFrame(image_rows)

        st.header("Portfolio analytics")
        total_objects = int(combined.shape[0])
        unique_images = int(image_summary.shape[0])
        categories = int(combined["object"].nunique()) if not combined.empty else 0
        avg_ms = float(image_summary["processing_time_ms"].mean()) if not image_summary.empty else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Images analyzed", unique_images)
        c2.metric("Objects detected", total_objects)
        c3.metric("Object categories", categories)
        c4.metric("Average processing time", f"{avg_ms:.0f} ms")

        if not combined.empty:
            summary = summarize_detections(combined)
            st.subheader("Detected objects by category")
            fig = px.bar(
                summary, x="object", y="count", text="count",
                title="Category-wise object count",
                labels={"object": "Object category", "count": "Detected count"},
            )
            fig.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.pie(
                summary, names="object", values="count",
                title="Share of detected objects",
            )
            st.plotly_chart(fig2, use_container_width=True)

            by_image = combine_image_summaries(combined)
            st.subheader("Object counts by image")
            fig3 = px.bar(
                by_image, x="image_name", y="count", color="object",
                title="Category counts across uploaded images",
                labels={"image_name": "Image", "count": "Count", "object": "Category"},
            )
            fig3.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("No detections passed the selected confidence threshold.")

        st.subheader("Image-level report")
        st.dataframe(image_summary, use_container_width=True, hide_index=True)

        st.subheader("Detection-level records")
        st.dataframe(combined, use_container_width=True, hide_index=True)

        st.download_button(
            "Download detection records (CSV)",
            data=combined.to_csv(index=False).encode("utf-8"),
            file_name="detection_records.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Download image summary (CSV)",
            data=image_summary.to_csv(index=False).encode("utf-8"),
            file_name="image_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.caption(
            "Counts are model predictions, not verified ground truth. Confidence is not the same as accuracy. "
            "Images are analyzed in memory; database storage is optional."
        )
