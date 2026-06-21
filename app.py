#============================================================
#   app.py  —  Medical Image Triage System
#   Streamlit Web Interface — Premium UI
#
#   FIXED: Line 851 — COVID-19 → COVID
#
#   CHALANE KA TARIKA:
#       streamlit run app.py
# ============================================================

import os
import cv2
import tempfile
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from keras.models import load_model
from keras.preprocessing import image as keras_image

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MediTriage AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PREMIUM CSS — Deep Navy + Emerald + Gold
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    /* ── Base ── */
    .stApp {
        background: linear-gradient(135deg, #030b14 0%, #050f1c 50%, #061018 100%);
        color: #c8dff0;
        font-family: 'Space Grotesk', sans-serif;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #040d18 0%, #061422 100%);
        border-right: 1px solid #0e2d4a;
    }
    [data-testid="stSidebar"] > div {
        padding-top: 0 !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: #040d18; }
    ::-webkit-scrollbar-thumb { background: #0e4d7a; border-radius: 10px; }

    /* ── Main Title ── */
    .hero-section {
        background: linear-gradient(135deg, #040d18, #061e33);
        border: 1px solid #0e3a5c;
        border-radius: 20px;
        padding: 35px 40px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 50%, #00d4ff08 0%, transparent 60%),
                    radial-gradient(circle at 70% 50%, #00ff9508 0%, transparent 60%);
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff, #00ff95, #00b4d8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
        margin: 0;
        line-height: 1.1;
    }
    .hero-subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #3a7fa5;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-top: 8px;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #00d4ff15, #00ff9515);
        border: 1px solid #00d4ff30;
        color: #00d4ff;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        padding: 4px 12px;
        border-radius: 20px;
        letter-spacing: 2px;
        margin-top: 12px;
    }

    /* ── Cards ── */
    .glass-card {
        background: linear-gradient(135deg, #061422cc, #040d18cc);
        border: 1px solid #0e2d4a;
        border-radius: 16px;
        padding: 22px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
        transition: border-color 0.3s ease;
    }
    .glass-card:hover {
        border-color: #1a4d7a;
    }

    /* ── Section Headers ── */
    .sec-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: #2a6a9a;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #0e2d4a;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ── Priority Badges ── */
    .priority-wrap {
        text-align: center;
        padding: 20px 0;
    }
    .badge-critical {
        background: linear-gradient(135deg, #4a0000, #8b0000, #cc0000);
        color: #ffcccc;
        padding: 12px 32px;
        border-radius: 50px;
        font-size: 1.05rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 3px;
        display: inline-block;
        box-shadow: 0 0 30px rgba(204, 0, 0, 0.35),
                    inset 0 1px 0 rgba(255,255,255,0.1);
        border: 1px solid #cc000040;
    }
    .badge-moderate {
        background: linear-gradient(135deg, #3a2800, #7a5500, #c47a00);
        color: #ffe0a0;
        padding: 12px 32px;
        border-radius: 50px;
        font-size: 1.05rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 3px;
        display: inline-block;
        box-shadow: 0 0 30px rgba(196, 122, 0, 0.35),
                    inset 0 1px 0 rgba(255,255,255,0.1);
        border: 1px solid #c47a0040;
    }
    .badge-normal {
        background: linear-gradient(135deg, #002a1a, #005533, #00804d);
        color: #a0ffd8;
        padding: 12px 32px;
        border-radius: 50px;
        font-size: 1.05rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 3px;
        display: inline-block;
        box-shadow: 0 0 30px rgba(0, 128, 77, 0.35),
                    inset 0 1px 0 rgba(255,255,255,0.1);
        border: 1px solid #00804d40;
    }

    /* ── Metric Values ── */
    .metric-big {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 600;
        line-height: 1;
        margin: 6px 0;
    }
    .metric-small {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: #2a6a9a;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    /* ── Confidence Bar ── */
    .conf-track {
        background: #0a1e2e;
        border-radius: 100px;
        height: 8px;
        width: 100%;
        margin-top: 10px;
        overflow: hidden;
        border: 1px solid #0e2d4a;
    }

    /* ── Detection Row ── */
    .detect-row {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 12px;
        background: #0a1e2e;
        border-radius: 8px;
        margin: 5px 0;
        border: 1px solid #0e2d4a;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #5aa5d5;
    }
    .detect-dot {
        width: 8px;
        height: 8px;
        background: #00d4ff;
        border-radius: 50%;
        flex-shrink: 0;
        box-shadow: 0 0 6px #00d4ff;
    }

    /* ── Sidebar Logo ── */
    .sidebar-logo {
        background: linear-gradient(135deg, #061422, #040d18);
        border-bottom: 1px solid #0e2d4a;
        padding: 28px 20px 20px;
        text-align: center;
        margin-bottom: 10px;
    }
    .sidebar-icon {
        font-size: 3rem;
        line-height: 1;
        filter: drop-shadow(0 0 12px #00d4ff50);
    }
    .sidebar-name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #c8dff0;
        margin-top: 8px;
        letter-spacing: -0.5px;
    }
    .sidebar-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem;
        color: #2a6a9a;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 4px;
    }

    /* ── Status Pills ── */
    .status-pill-ok {
        display: flex;
        align-items: center;
        gap: 10px;
        background: #002a1a;
        border: 1px solid #00804d30;
        border-radius: 10px;
        padding: 10px 14px;
        margin: 6px 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #00c47a;
    }
    .status-pill-err {
        display: flex;
        align-items: center;
        gap: 10px;
        background: #2a0000;
        border: 1px solid #cc000030;
        border-radius: 10px;
        padding: 10px 14px;
        margin: 6px 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #ff6b6b;
    }
    .status-dot-ok {
        width: 8px; height: 8px;
        background: #00c47a;
        border-radius: 50%;
        box-shadow: 0 0 8px #00c47a;
        flex-shrink: 0;
    }
    .status-dot-err {
        width: 8px; height: 8px;
        background: #ff4444;
        border-radius: 50%;
        box-shadow: 0 0 8px #ff4444;
        flex-shrink: 0;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }

    /* ── Analyze Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #003a4a, #005566);
        color: #00d4ff;
        border: 1px solid #00d4ff30;
        border-radius: 12px;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 3px;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 14px;
        width: 100%;
        transition: all 0.3s ease;
        text-transform: uppercase;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #005566, #007788);
        border-color: #00d4ff80;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.25),
                    0 4px 15px rgba(0, 0, 0, 0.3);
        transform: translateY(-1px);
    }
    .stButton > button:active {
        transform: translateY(0px);
    }

    /* ── Upload Box ── */
    [data-testid="stFileUploader"] {
        background: #040d18;
        border: 2px dashed #0e3a5c;
        border-radius: 16px;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #00d4ff40;
    }

    /* ── Slider ── */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #00d4ff, #00ff95) !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: #00d4ff !important;
    }

    /* ── Info/Error boxes ── */
    .stAlert {
        background: #040d18;
        border-radius: 12px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
    }

    /* ── How it works cards ── */
    .step-card {
        background: linear-gradient(135deg, #061422, #040d18);
        border: 1px solid #0e2d4a;
        border-radius: 16px;
        padding: 24px 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .step-card:hover {
        border-color: #00d4ff30;
        box-shadow: 0 8px 30px rgba(0, 212, 255, 0.08);
        transform: translateY(-2px);
    }
    .step-num {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff10, #00ff9510);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
    }
    .step-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        color: #00d4ff;
        margin-top: 6px;
        letter-spacing: 1px;
    }
    .step-desc {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #2a6a9a;
        margin-top: 8px;
        line-height: 1.6;
    }

    /* ── Empty state ── */
    .empty-state {
        text-align: center;
        padding: 70px 20px;
        border: 1px dashed #0e2d4a;
        border-radius: 20px;
        margin: 20px 0;
        background: linear-gradient(135deg, #040d1880, #061422 80%);
    }
    .empty-icon {
        font-size: 5rem;
        filter: drop-shadow(0 0 20px #00d4ff30);
        margin-bottom: 16px;
    }
    .empty-text {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        color: #2a6a9a;
        letter-spacing: 1px;
    }
    .empty-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: #1a3a5a;
        margin-top: 8px;
        letter-spacing: 2px;
    }

    /* ── Image caption ── */
    .image-caption {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: #2a6a9a;
        text-align: center;
        margin-top: 6px;
        letter-spacing: 1px;
    }

    /* ── Divider ── */
    hr {
        border: none;
        border-top: 1px solid #0e2d4a;
        margin: 20px 0;
    }

    /* ── Hide Streamlit UI (SAFE) ── */
    #MainMenu, footer {
        visibility: hidden;
    }

    .block-container {
        padding-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT= os.path.dirname(os.path.abspath(__file__))

TRANSFER_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "classification",
    "transfer_model_resnet50.h5"
)

YOLO_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "detection",
    "best.pt"
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_cnn_model():
    if not os.path.exists(TRANSFER_MODEL_PATH):
        return None
    return load_model(TRANSFER_MODEL_PATH)

@st.cache_resource
def load_yolo_model():
    if not os.path.exists(YOLO_MODEL_PATH):
        return None
    return YOLO(YOLO_MODEL_PATH)

# ============================================================
# HELPERS
# ============================================================

CLASS_NAMES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

def get_priority(conf):
    if conf >= 0.7:
        return "Critical", "badge-critical", "🔴", "#cc2200"
    elif conf >= 0.4:
        return "Moderate", "badge-moderate", "🟡", "#c47a00"
    else:
        return "Normal",   "badge-normal",   "🟢", "#00804d"


# ================= CNN PREDICTION (UPDATED FOR 4 CLASSES) =================
def cnn_predict(img, model):
    img_r = img.resize((224, 224))
    x     = keras_image.img_to_array(img_r) / 255.0
    x     = np.expand_dims(x, axis=0)

    preds = model.predict(x, verbose=0)[0]   # Softmax output (4 values)
    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])

    return CLASS_NAMES[class_idx], confidence


# ================= YOLO DETECTION (UPDATED FOR 4 CLASSES) =================
def yolo_detect(img, model):
    img_np  = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        temp_path = tmp.name
        cv2.imwrite(temp_path, img_bgr)

    results    = model.predict(source=temp_path, imgsz=640, conf=0.4, verbose=False)
    img_draw   = img_bgr.copy()
    cls_names  = model.names
    detections = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls   = result.boxes.cls.cpu().numpy()

        for (x1, y1, x2, y2), c, cl in zip(boxes, confs, cls):
            idx  = int(cl)
            name = cls_names[idx] if idx < len(cls_names) else str(idx)

            if name == "Normal":
                color = (0, 220, 120)
            else:
                color = (50, 80, 255)

            cv2.rectangle(img_draw,
                          (int(x1), int(y1)), (int(x2), int(y2)),
                          color, 2)

            label = f"{name}  {c:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)

            cv2.rectangle(img_draw,
                          (int(x1), int(y1) - th - 12),
                          (int(x1) + tw + 10, int(y1)),
                          color, -1)

            cv2.putText(img_draw, label,
                        (int(x1) + 5, int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2)

            detections.append({"class": name, "conf": float(c)})

    try:
        os.remove(temp_path)
    except Exception:
        pass

    return Image.fromarray(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)), detections


# ============================================================
# SIDEBAR
# ============================================================

cnn_model  = load_cnn_model()
yolo_model = load_yolo_model()

with st.sidebar:

    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-icon">🫁</div>
        <div class="sidebar-name">MediTriage AI</div>
        <div class="sidebar-tag">v3.0 — 4-Class Lung Detection</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-header">⬡ System Status</div>',
                unsafe_allow_html=True)

    if cnn_model:
        st.markdown("""
        <div class="status-pill-ok">
            <div class="status-dot-ok"></div>
            CNN ResNet50 — Online
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-pill-err">
            <div class="status-dot-err"></div>
            CNN Model — Not Found
        </div>""", unsafe_allow_html=True)

    if yolo_model:
        st.markdown("""
        <div class="status-pill-ok">
            <div class="status-dot-ok"></div>
            YOLOv8n — Online
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-pill-err">
            <div class="status-dot-err"></div>
            YOLO — Run Pipeline First
        </div>""", unsafe_allow_html=True)
        st.code("python src/yolo_full_pipeline.py", language="bash")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="sec-header">⬡ Detection Settings</div>',
                unsafe_allow_html=True)

    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.4, 0.05)
    show_boxes = st.checkbox("Show YOLO Bounding Boxes", value=True)
    show_conf  = st.checkbox("Show Confidence Details",  value=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="sec-header">⬡ Model Info</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family: JetBrains Mono, monospace; font-size: 0.7rem;
                color: #2a6a9a; line-height: 2.2;'>
        Classifier &nbsp;→&nbsp; ResNet50<br>
        Detector &nbsp;&nbsp;→&nbsp; YOLOv8 Nano<br>
        Classes &nbsp;&nbsp;&nbsp;→&nbsp; COVID / Lung_Opacity / Normal / Viral Pneumonia<br>
        Input &nbsp;&nbsp;&nbsp;&nbsp;→&nbsp; Chest X-Ray<br>
        Framework &nbsp;→&nbsp; TensorFlow + PT
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# MAIN
# ============================================================

st.markdown("""
<div class="hero-section">
    <div class="hero-title">Medical Image Triage</div>
    <div class="hero-subtitle">AI-Powered Chest X-Ray Analysis System</div>
    <div class="hero-badge">● CNN + YOLO DETECTION ENGINE</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="sec-header">⬡ Upload Chest X-Ray</div>',
            unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop your X-Ray here",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1.1, 0.9], gap="large")

    with col1:
        st.markdown('<div class="sec-header">⬡ Input Image</div>',
                    unsafe_allow_html=True)

        st.image(img, use_container_width=True)
        st.markdown(f"""
        <div class="image-caption">
            {uploaded_file.name} &nbsp;|&nbsp;
            {img.width} × {img.height} px &nbsp;|&nbsp;
            {uploaded_file.size / 1024:.1f} KB
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="sec-header">⬡ Analysis</div>',
                    unsafe_allow_html=True)

        if not cnn_model:
            st.error("CNN model not found!")
        elif not yolo_model:
            st.error("YOLO model not found! Run pipeline first.")
        else:
            analyze = st.button("⬡  RUN ANALYSIS", use_container_width=True)

            if analyze:
                with st.spinner("Running AI inference..."):

                    pred_class, cnn_conf = cnn_predict(img, cnn_model)

                    priority, badge, icon, color = get_priority(cnn_conf)
                    yolo_img, detections  = yolo_detect(img, yolo_model)

                st.session_state["results"] = {
                    "pred_class": pred_class,
                    "cnn_conf"  : cnn_conf,
                    "priority"  : priority,
                    "badge"     : badge,
                    "icon"      : icon,
                    "color"     : color,
                    "detections": detections,
                    "yolo_img"  : yolo_img
                }

            if "results" in st.session_state:
                r = st.session_state["results"]

                st.markdown(f"""
                <div class="glass-card priority-wrap">
                    <div class="metric-small" style="margin-bottom:14px;">
                        Triage Priority
                    </div>
                    <span class="{r['badge']}">
                        {r['icon']} &nbsp; {r['priority'].upper()}
                    </span>
                </div>
                """, unsafe_allow_html=True)

                if show_conf:
                    conf_pct = int(r["cnn_conf"] * 100)
                    st.markdown(f"""
                    <div class="glass-card">

                    <div class="metric-small">Predicted Disease</div>
                    <div class="metric-big" style="color:{r['color']}">
                    {r['pred_class']}
                    </div>

                    <div class="metric-small" style="margin-top:12px;">
                    CNN Confidence
                    </div>

                    <div class="metric-big" style="color:{r['color']}">
                    {r['cnn_conf']:.4f}
                    </div>

                    <div class="conf-track">
                    <div style="background: linear-gradient(90deg, {r['color']}, {r['color']}88);
                    width:{int(r["cnn_conf"]*100)}%;
                    height:8px;
                    border-radius:100px;">
                    </div>
                    </div>

                    <div class="metric-small" style="margin-top:6px;">
                    {int(r["cnn_conf"]*100)}% — {r['priority']}
                     </div>

                    </div>
                    """, unsafe_allow_html=True)
                det_count = len(r["detections"])
                st.markdown(f"""
                <div class="glass-card">
                <div class="metric-small">YOLO Objects Found</div>
                <div class="metric-big" style="color:#00d4ff">
                {det_count}
                 </div>
                 </div>
                 """, unsafe_allow_html=True)

                if r["detections"]:
                    for d in r["detections"]:
                        st.markdown(f"""
                        <div class="detect-row">
                            <div class="detect-dot"></div>
                            {d['class']} &nbsp;—&nbsp; {d['conf']:.0%} confidence
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style='font-family: JetBrains Mono, monospace;
                                font-size: 0.72rem; color: #1a4a6a;
                                padding: 6px 0;'>
                        No detections above threshold
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("""
                <div class="glass-card">
                    <div class="metric-small" style="margin-bottom:10px;">
                        Priority Reference
                    </div>
                    <div style='font-family: JetBrains Mono, monospace;
                                font-size: 0.72rem; line-height: 2.2;'>
                        <span style='color:#cc2200'>🔴 Critical</span>
                        <span style='color:#1a4a6a'> conf ≥ 0.70</span><br>
                        <span style='color:#c47a00'>🟡 Moderate</span>
                        <span style='color:#1a4a6a'> conf ≥ 0.40</span><br>
                        <span style='color:#00804d'>🟢 Normal &nbsp;</span>
                        <span style='color:#1a4a6a'> conf &lt; 0.40</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)


    if show_boxes and "results" in st.session_state:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="sec-header">⬡ YOLO Detection Output</div>',
                    unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.image(img, caption="Original X-Ray", use_container_width=True)
        with c2:
            st.image(st.session_state["results"]["yolo_img"],
                     caption="YOLO Annotated", use_container_width=True)

if "results" in st.session_state:
    r = st.session_state["results"]

    priority   = r["priority"]
    pred_class = r["pred_class"]

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sec-header">⬡ Patient Prediction Report</div>',
                unsafe_allow_html=True)

    # FIXED: COVID-19 → COVID
    if pred_class == "COVID":

        if priority == "Critical":
            title = "COVID-19 — High Probability Detected"
        else:
            title = "COVID-19 — Moderate Signs Detected"

        title_color = "#ff4444" if priority == "Critical" else "#ffaa00"
        icon        = "🔴" if priority == "Critical" else "🟡"
        card_border = "#cc2200" if priority == "Critical" else "#c47a00"

        description = [
            "**What is COVID-19?**",
            "COVID-19 is a respiratory infection affecting the lungs.",
            "The AI model detected radiological patterns consistent with COVID-related pneumonia.",
            "",
            "**Common Symptoms:**",
            "- 🌡️ Fever",
            "- 😮‍💨 Persistent cough",
            "- 🫁 Shortness of breath",
            "- 😔 Fatigue",
            "- 👃 Loss of smell or taste",
        ]

        advice = [
            "**⚠️ Medical Consultation Recommended**",
            "- ✅ Visit a healthcare provider",
            "- ✅ Get RT-PCR test confirmation",
            "- ✅ Monitor oxygen saturation",
            "- ✅ Isolate if symptomatic",
            "- ❌ Do not ignore breathing difficulty",
        ]

    elif pred_class == "Viral Pneumonia":

        title       = "Viral Pneumonia Signs Detected"
        title_color = "#ffaa00"
        icon        = "🟡"
        card_border = "#c47a00"

        description = [
            "**What is Viral Pneumonia?**",
            "Viral pneumonia is a lung infection caused by viruses.",
            "X-Ray shows inflammatory patterns consistent with viral infection.",
            "",
            "**Possible Symptoms:**",
            "- 🌡️ Fever",
            "- 🤧 Cough",
            "- 🫁 Chest discomfort",
            "- 😔 Fatigue",
        ]

        advice = [
            "**⚠️ Doctor Consultation Advised**",
            "- ✅ Consult physician",
            "- ✅ Take prescribed antivirals if needed",
            "- ✅ Rest and hydration",
            "- ❌ Avoid self-medication",
        ]

    elif pred_class == "Lung Opacity" or pred_class == "Lung_Opacity":

        title       = "Lung Opacity Detected"
        title_color = "#ff8800"
        icon        = "🟠"
        card_border = "#cc7700"

        description = [
            "**What is Lung Opacity?**",
            "Lung opacity refers to white patches seen in X-Ray.",
            "It may indicate infection, inflammation, or fluid accumulation.",
            "",
            "**Further evaluation required.**"
        ]

        advice = [
            "**⚠️ Further Tests Recommended**",
            "- ✅ CT Scan if advised",
            "- ✅ Blood tests",
            "- ✅ Clinical examination",
            "- ❌ Do not ignore symptoms",
        ]

    else:  # Normal

        title       = "No Significant Abnormality Detected"
        title_color = "#00cc77"
        icon        = "🟢"
        card_border = "#00804d"

        description = [
            "**Normal Chest X-Ray**",
            "No major abnormalities detected in lung fields.",
            "",
            "**This suggests:**",
            "- ✔️ Clear lung fields",
            "- ✔️ No visible pneumonia patterns",
        ]

        advice = [
            "**✅ No Immediate Concern**",
            "- ✅ Maintain healthy lifestyle",
            "- ✅ Consult doctor if symptoms persist",
            "- ℹ️ AI screening is not a final diagnosis",
        ]

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#061422cc,#040d18cc);
                    border:1px solid {card_border}40; border-radius:16px;
                    padding:22px; margin:10px 0;'>
            <div class="metric-small" style="margin-bottom:12px;">
                Disease Information
            </div>
            <div style='font-size:1rem; font-weight:700; color:{title_color};
                        margin-bottom:16px;'>
                {icon} &nbsp; {title}
            </div>
        """, unsafe_allow_html=True)

        for line in description:
            st.markdown(line)

        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#061422cc,#040d18cc);
                    border:1px solid {card_border}40; border-radius:16px;
                    padding:22px; margin:10px 0;'>
            <div class="metric-small" style="margin-bottom:12px;">
                Doctor's Advice
            </div>
        """, unsafe_allow_html=True)

        for line in advice:
            st.markdown(line)

        st.markdown("</div>", unsafe_allow_html=True)

    st.info("⚠️ DISCLAIMER: This AI analysis is for screening purposes only. Always consult a qualified healthcare professional.")

else:
    if "results" in st.session_state:
        del st.session_state["results"]

    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🫁</div>
        <div class="empty-text">Upload a Chest X-Ray to Begin</div>
        <div class="empty-sub">JPG · JPEG · PNG · WEBP supported</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sec-header">⬡ How It Works</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    steps = [
        ("01", "Upload", "Chest X-Ray image\nupload karo"),
        ("02", "Analyze", "CNN + YOLO dono\nse analysis hoti hai"),
        ("03", "Result",  "Priority triage\nresult milta hai"),
    ]
    for col, (num, title, desc) in zip([c1, c2, c3], steps):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-num">{num}</div>
                <div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
