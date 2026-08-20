"""
YoloTrackX — Multi-class Object Detection & Tracking with YOLOv8
Beautiful Streamlit UI with burgundy / gold / cream theme.
"""

from __future__ import annotations

import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from utils.helper import (
    count_classes,
    draw_detections,
    draw_stats_overlay,
    extract_detections_table,
    get_available_models,
    load_model,
    resize_frame,
)

# ---------------------------------------------------------------------------
# Page config & custom CSS (burgundy / gold / cream)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="YoloTrackX | Object Detection & Tracking",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --burgundy: #800020;
    --burgundy-dark: #5c0018;
    --burgundy-light: #a33a52;
    --gold: #d4af37;
    --gold-light: #f0d77b;
    --cream: #fff8f0;
    --cream-dark: #f5e6d3;
}

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

.stApp {
    background: linear-gradient(160deg, #fff8f0 0%, #f5e6d3 40%, #f0d5c8 100%);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #800020 0%, #5c0018 100%) !important;
}
section[data-testid="stSidebar"] * {
    color: #fff8f0 !important;
}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stMultiSelect label {
    color: #f0d77b !important;
    font-weight: 500;
}

h1 {
    color: #800020 !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}
h2, h3 {
    color: #5c0018 !important;
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #800020 0%, #a33a52 100%);
    border-radius: 12px;
    padding: 12px 16px;
    box-shadow: 0 4px 15px rgba(128, 0, 32, 0.25);
}
div[data-testid="stMetric"] label {
    color: #f0d77b !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #fff8f0 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #800020 0%, #a33a52 100%) !important;
    color: #fff8f0 !important;
    border: 2px solid #d4af37 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.25s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #d4af37 0%, #f0d77b 100%) !important;
    color: #5c0018 !important;
    border-color: #800020 !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(212, 175, 55, 0.4);
}

section[data-testid="stFileUploader"] {
    background: rgba(128, 0, 32, 0.06);
    border: 2px dashed #d4af37;
    border-radius: 12px;
    padding: 8px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background: #fff8f0;
    border-radius: 8px 8px 0 0;
    color: #800020;
    font-weight: 500;
    border: 1px solid #d4af37;
}
.stTabs [aria-selected="true"] {
    background: #800020 !important;
    color: #fff8f0 !important;
}

.streamlit-expanderHeader {
    background: rgba(128, 0, 32, 0.08);
    border-radius: 8px;
    color: #5c0018 !important;
    font-weight: 600;
}

.footer {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
    color: #800020;
    font-size: 0.9rem;
    opacity: 0.85;
}
.footer a {
    color: #d4af37;
    text-decoration: none;
    font-weight: 600;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def init_session():
    defaults = {
        "model": None,
        "model_name": None,
        "all_detections": [],
        "unique_track_ids": set(),
        "processed_video_path": None,
        "last_fps": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


def render_sidebar():
    st.sidebar.markdown("## 🎯 YoloTrackX")
    st.sidebar.markdown("**Multi-class Detection & Tracking**")
    st.sidebar.markdown("---")

    models = get_available_models()
    model_label = st.sidebar.selectbox(
        "🧠 Model",
        list(models.keys()),
        index=0,
        help="Nano is fastest; XLarge is most accurate.",
    )
    model_path = models[model_label]

    if st.session_state.model_name != model_path:
        st.session_state.model = None

    if st.session_state.model is None:
        st.sidebar.info("Load a YOLO model to enable detection.")
        if st.sidebar.button("Load model", type="primary"):
            with st.spinner(f"Loading {model_label}…"):
                st.session_state.model = load_model(model_path)
                st.session_state.model_name = model_path
            st.rerun()
        return None

    conf = st.sidebar.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.35,
        step=0.05,
    )
    iou = st.sidebar.slider(
        "IoU threshold (NMS)",
        min_value=0.1,
        max_value=0.9,
        value=0.45,
        step=0.05,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Tracking")
    enable_tracking = st.sidebar.checkbox("Enable multi-object tracking", value=True)
    tracker = st.sidebar.selectbox(
        "Tracker algorithm",
        ["bytetrack.yaml", "botsort.yaml"],
        index=0,
        help="ByteTrack is faster; BoT-SORT is more robust to occlusion.",
        disabled=not enable_tracking,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎨 Display options")
    show_labels = st.sidebar.checkbox("Show class labels", value=True)
    show_conf = st.sidebar.checkbox("Show confidence", value=True)
    show_ids = st.sidebar.checkbox("Show track IDs", value=True)
    show_stats = st.sidebar.checkbox("Show live stats overlay", value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📦 Class filter")
    model = st.session_state.model
    all_classes = list(model.names.values())
    selected_classes = st.sidebar.multiselect(
        "Detect only these classes (empty = all)",
        options=all_classes,
        default=[],
        help="Leave empty to detect all 80 COCO classes.",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Made with ❤️ · YOLOv8 + Streamlit")

    return {
        "model": model,
        "conf": conf,
        "iou": iou,
        "enable_tracking": enable_tracking,
        "tracker": tracker,
        "show_labels": show_labels,
        "show_conf": show_conf,
        "show_ids": show_ids,
        "show_stats": show_stats,
        "selected_classes": selected_classes,
        "class_id_map": {v: k for k, v in model.names.items()},
    }


def filter_result_by_classes(result, selected_classes: List[str], class_id_map: Dict):
    if not selected_classes or result.boxes is None or len(result.boxes) == 0:
        return result

    keep_ids = {class_id_map[c] for c in selected_classes if c in class_id_map}
    mask = np.isin(result.boxes.cls.cpu().numpy().astype(int), list(keep_ids))
    if not mask.any():
        result.boxes = None
        return result

    result.boxes = result.boxes[mask]
    return result


def run_image_inference(image: np.ndarray, cfg: dict):
    model: YOLO = cfg["model"]
    conf, iou = cfg["conf"], cfg["iou"]

    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        verbose=False,
    )
    result = results[0]
    result = filter_result_by_classes(result, cfg["selected_classes"], cfg["class_id_map"])

    annotated = draw_detections(
        image,
        result,
        show_labels=cfg["show_labels"],
        show_conf=cfg["show_conf"],
        show_track_id=False,
    )
    class_counts = count_classes(result)
    table = extract_detections_table(result)
    return annotated, class_counts, table, result


def process_video(
    video_path: str,
    cfg: dict,
    progress_bar,
    status_text,
    frame_placeholder,
    max_frames: Optional[int] = None,
):
    model: YOLO = cfg["model"]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Could not open video.")
        return None, [], set()

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps_src, (width, height))

    all_rows: List[Dict] = []
    unique_ids: Set[int] = set()
    frame_idx = 0
    t0 = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break

        frame_idx += 1
        if cfg["enable_tracking"]:
            results = model.track(
                source=frame,
                conf=cfg["conf"],
                iou=cfg["iou"],
                persist=True,
                tracker=cfg["tracker"],
                verbose=False,
            )
        else:
            results = model.predict(
                source=frame,
                conf=cfg["conf"],
                iou=cfg["iou"],
                verbose=False,
            )

        result = results[0]
        result = filter_result_by_classes(
            result, cfg["selected_classes"], cfg["class_id_map"]
        )

        annotated = draw_detections(
            frame,
            result,
            show_labels=cfg["show_labels"],
            show_conf=cfg["show_conf"],
            show_track_id=cfg["show_ids"] and cfg["enable_tracking"],
        )

        class_counts = count_classes(result)
        total_obj = sum(class_counts.values())

        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy().astype(int)
            unique_ids.update(ids.tolist())

        elapsed = time.time() - t0
        fps = frame_idx / elapsed if elapsed > 0 else 0.0

        if cfg["show_stats"]:
            annotated = draw_stats_overlay(
                annotated,
                fps=fps,
                total_objects=total_obj,
                class_counts=class_counts,
                unique_tracks=len(unique_ids),
            )

        writer.write(annotated)

        if frame_idx % 3 == 0 or frame_idx == 1:
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb, channels="RGB", use_container_width=True)
            progress = min(frame_idx / max(total_frames, 1), 1.0)
            progress_bar.progress(progress)
            status_text.markdown(
                f"**Frame** {frame_idx}/{total_frames} · "
                f"**FPS** {fps:.1f} · "
                f"**Objects** {total_obj} · "
                f"**Unique tracks** {len(unique_ids)}"
            )

        rows = extract_detections_table(result)
        for r in rows:
            r["frame"] = frame_idx
            all_rows.append(r)

    cap.release()
    writer.release()
    st.session_state.last_fps = fps if frame_idx else 0.0
    return out_path, all_rows, unique_ids


def render_header():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🎯 YoloTrackX")
        st.markdown(
            "#### Multi-class Object Detection & Tracking powered by **YOLOv8**"
        )
        st.markdown(
            "Upload an image or video, or use your webcam. "
            "Track objects across frames with ByteTrack / BoT-SORT, "
            "filter classes, and export results."
        )
    with col2:
        st.markdown("")
        st.markdown(
            """
            <div style="background:linear-gradient(135deg,#800020,#a33a52);
                        border-radius:16px;padding:1.2rem;text-align:center;
                        border:2px solid #d4af37;margin-top:1rem;">
                <div style="font-size:2rem;">🚀</div>
                <div style="color:#f0d77b;font-weight:600;">Real-time ready</div>
                <div style="color:#fff8f0;font-size:0.85rem;">80+ COCO classes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_metrics(class_counts: Dict[str, int], unique_tracks: int = 0, fps: float = 0):
    total = sum(class_counts.values())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Detected objects", total)
    c2.metric("Unique classes", len(class_counts))
    c3.metric("Unique tracks", unique_tracks if unique_tracks else "—")
    c4.metric("Processing FPS", f"{fps:.1f}" if fps else "—")


def render_class_chart(class_counts: Dict[str, int]):
    if not class_counts:
        return
    df = pd.DataFrame(
        {"Class": list(class_counts.keys()), "Count": list(class_counts.values())}
    ).sort_values("Count", ascending=True)
    fig = px.bar(
        df,
        x="Count",
        y="Class",
        orientation="h",
        color="Count",
        color_continuous_scale=["#f0d77b", "#d4af37", "#800020"],
        title="Objects per class",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="Outfit",
        title_font_color="#5c0018",
        margin=dict(l=10, r=10, t=40, b=10),
        height=max(220, len(class_counts) * 28 + 80),
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    render_header()
    cfg = render_sidebar()
    if cfg is None:
        st.info("Use the sidebar to load a YOLO model before starting detection.")
        return

    tab_image, tab_video, tab_webcam, tab_about = st.tabs(
        ["🖼️ Image", "🎬 Video", "📷 Webcam", "ℹ️ About"]
    )

    # ------------------------------------------------------------------ Image
    with tab_image:
        st.subheader("Image Detection")
        uploaded = st.file_uploader(
            "Upload an image (JPG, PNG, WEBP, BMP)",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="img_uploader",
        )

        col_a, col_b = st.columns(2)
        if uploaded is not None:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = resize_frame(image, max_side=1280)

            with col_a:
                st.markdown("**Original**")
                st.image(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )

            if st.button("🚀 Detect Objects", key="btn_img"):
                with st.spinner("Running YOLOv8…"):
                    annotated, class_counts, table, _ = run_image_inference(image, cfg)

                with col_b:
                    st.markdown("**Detections**")
                    st.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        use_container_width=True,
                    )

                render_metrics(class_counts)
                render_class_chart(class_counts)

                if table:
                    df = pd.DataFrame(table)
                    with st.expander("📋 Detection details", expanded=False):
                        st.dataframe(df, use_container_width=True)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download CSV",
                        data=csv,
                        file_name="yolotrackx_detections.csv",
                        mime="text/csv",
                    )

                _, buf = cv2.imencode(".png", annotated)
                st.download_button(
                    "⬇️ Download annotated image",
                    data=buf.tobytes(),
                    file_name="yolotrackx_result.png",
                    mime="image/png",
                )
        else:
            st.info("👆 Upload an image to get started.")

    # ------------------------------------------------------------------ Video
    with tab_video:
        st.subheader("Video Detection & Tracking")
        video_file = st.file_uploader(
            "Upload a video (MP4, AVI, MOV, MKV)",
            type=["mp4", "avi", "mov", "mkv"],
            key="vid_uploader",
        )

        max_frames = st.number_input(
            "Max frames to process (0 = all)",
            min_value=0,
            max_value=10000,
            value=0,
            step=50,
            help="Useful for long videos or quick tests.",
        )

        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            tfile.close()
            video_path = tfile.name

            st.video(video_path)

            if st.button("🚀 Process Video", key="btn_vid"):
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                frame_placeholder = st.empty()

                with st.spinner("Processing video… this may take a while"):
                    out_path, all_rows, unique_ids = process_video(
                        video_path,
                        cfg,
                        progress_bar,
                        status_text,
                        frame_placeholder,
                        max_frames=max_frames if max_frames > 0 else None,
                    )

                if out_path:
                    st.success("✅ Processing complete!")
                    st.session_state.processed_video_path = out_path
                    st.session_state.all_detections = all_rows
                    st.session_state.unique_track_ids = unique_ids

                    class_counts: Dict[str, int] = defaultdict(int)
                    for r in all_rows:
                        class_counts[r["class"]] += 1

                    render_metrics(
                        dict(class_counts),
                        unique_tracks=len(unique_ids),
                        fps=st.session_state.last_fps,
                    )
                    render_class_chart(dict(class_counts))

                    st.markdown("### 🎥 Result preview")
                    try:
                        with open(out_path, "rb") as f:
                            st.download_button(
                                "⬇️ Download processed video",
                                data=f.read(),
                                file_name="yolotrackx_tracked.mp4",
                                mime="video/mp4",
                            )
                    except Exception as e:
                        st.warning(f"Could not prepare video download: {e}")

                    if all_rows:
                        df = pd.DataFrame(all_rows)
                        with st.expander("📋 All detections (frame-level)", expanded=False):
                            st.dataframe(df.head(500), use_container_width=True)
                            st.caption(f"Showing first 500 of {len(df)} rows")
                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ Download full CSV",
                            data=csv,
                            file_name="yolotrackx_video_detections.csv",
                            mime="text/csv",
                        )
        else:
            st.info("👆 Upload a video to run detection & tracking.")

    # ------------------------------------------------------------------ Webcam
    with tab_webcam:
        st.subheader("Webcam (local)")
        st.markdown(
            "Click **Start** to open your default webcam. "
            "Press **Stop** when finished. Tracking works across consecutive frames."
        )
        st.warning(
            "Webcam access works best when running Streamlit **locally**. "
            "On Streamlit Cloud the browser cannot access your camera from the server."
        )

        run_cam = st.checkbox("Start webcam", value=False, key="cam_toggle")
        cam_placeholder = st.empty()
        cam_metrics = st.empty()

        if run_cam:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open webcam. Make sure a camera is connected.")
            else:
                unique_ids: Set[int] = set()
                t0 = time.time()
                frame_count = 0
                stop = st.button("⏹ Stop webcam", key="stop_cam")

                while run_cam and not stop:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read from webcam.")
                        break

                    frame = resize_frame(frame, max_side=960)
                    frame_count += 1

                    if cfg["enable_tracking"]:
                        results = cfg["model"].track(
                            source=frame,
                            conf=cfg["conf"],
                            iou=cfg["iou"],
                            persist=True,
                            tracker=cfg["tracker"],
                            verbose=False,
                        )
                    else:
                        results = cfg["model"].predict(
                            source=frame,
                            conf=cfg["conf"],
                            iou=cfg["iou"],
                            verbose=False,
                        )

                    result = results[0]
                    result = filter_result_by_classes(
                        result, cfg["selected_classes"], cfg["class_id_map"]
                    )

                    annotated = draw_detections(
                        frame,
                        result,
                        show_labels=cfg["show_labels"],
                        show_conf=cfg["show_conf"],
                        show_track_id=cfg["show_ids"] and cfg["enable_tracking"],
                    )


if __name__ == "__main__":
    main()
