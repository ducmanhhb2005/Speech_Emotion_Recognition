import os
import tempfile
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import soundfile as sf
import torch
import torch.nn.functional as F

from utils.loaders import (
    load_label_encoder,
    load_ml_model,
    load_scaler,
    load_keras_model,
    load_torch_model,
)
from utils.preprocess import (
    get_features_cnn_lstm,
    get_features_ml,
    get_features_cnn,
    prepare_input_cnn_lstm,
    prepare_input_ml,
    prepare_input_cnn,
)


ART_DIR = "artifacts"

MODEL_REGISTRY = {
    "CNN-LSTM (ZCR + RMSE + MFCC flatten)": {
        "kind": "cnn_lstm",
        "model_path": os.path.join(ART_DIR, "cnn_lstm", "model.h5"),
        "scaler_path": os.path.join(ART_DIR, "cnn_lstm", "scaler.pkl"),
        "get_features": get_features_cnn_lstm,
        "prepare_input": prepare_input_cnn_lstm,
    },
    "ML (XGBOOST)": {
        "kind": "ml",
        "model_path": os.path.join(ART_DIR, "ml", "model.pkl"),
        "scaler_path": os.path.join(ART_DIR, "ml", "scaler.pkl"),  # nếu không có thì để None
        "get_features": get_features_ml,
        "prepare_input": prepare_input_ml,
    },
    "CNN": {
        "kind": "cnn",
        "model_path": os.path.join(ART_DIR, "cnn", "model.pth"),
        "scaler_path": os.path.join(ART_DIR, "cnn", "scaler.pkl"),  # nếu không có thì để None
        "get_features": get_features_cnn,
        "prepare_input": prepare_input_cnn,
    },
}

LABEL_ENCODER_PATH = os.path.join(ART_DIR, "label_encoder.pkl")


@st.cache_resource
def load_all():
    le = load_label_encoder(LABEL_ENCODER_PATH)

    cache = {}
    for name, cfg in MODEL_REGISTRY.items():
        kind = cfg["kind"]
        model_path = cfg["model_path"]
        scaler_path = cfg.get("scaler_path")

        if kind == "ml":
            model = load_ml_model(model_path)
        elif kind == "cnn":
            model = load_torch_model(model_path)
        else:  # cnn_lstm
            model = load_keras_model(model_path)

        scaler = None
        if scaler_path and os.path.exists(scaler_path):
            scaler = load_scaler(scaler_path)

        cache[name] = {
            "kind": kind,
            "model": model,
            "scaler": scaler,
            "get_features": cfg["get_features"],
            "prepare_input": cfg["prepare_input"],
        }

    return le, cache


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)


def predict(model_kind, model, x, le):

    if model_kind == "ml":
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x)[0]
        elif hasattr(model, "decision_function"):
            scores = np.ravel(model.decision_function(x))
            probs = softmax(scores)
        else:
            pred = int(np.ravel(model.predict(x))[0])
            probs = np.zeros(len(le.classes_), dtype=np.float32)
            probs[pred] = 1.0

    elif model_kind == "cnn_lstm":
        probs = model.predict(x, verbose=0)[0]

    elif model_kind == "cnn":
        x_tensor = torch.from_numpy(x.astype(np.float32)).unsqueeze(1)  # (1, 1, T, F)
        model.eval()
        with torch.no_grad():
            outputs = model(x_tensor)           # logits
            probs = F.softmax(outputs, dim=1)   # xác suất
            probs = probs[0].detach().cpu().numpy()  # lấy xác suất của sample đầu tiên

    # Lấy nhãn dự đoán chung cho tất cả loại model
    probs = np.asarray(probs, dtype=np.float32)
    idx = int(np.argmax(probs))
    label = le.inverse_transform([idx])[0]
    return label, probs

audio_path = None

st.set_page_config(page_title="SRE/SER Deploy", layout="centered")
st.title("🎙️ SRE/SER Deploy (Upload audio → chọn model → kết quả)")

le, models_cache = load_all()

model_name = st.selectbox("Chọn model", list(MODEL_REGISTRY.keys()))
import streamlit as st
import sounddevice as sd
import numpy as np
import tempfile
import os
from scipy.io.wavfile import write

# --- Khởi tạo session_state cho audio_path ---
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None

audio_source = st.radio("Chọn nguồn audio", ["Upload file", "Nói trực tiếp (Microphone)"])

# ------------------ Upload file ------------------
if audio_source == "Upload file":
    uploaded = st.file_uploader("Upload audio", type=["wav", "mp3", "flac", "ogg", "m4a"])
    if uploaded is not None:
        st.audio(uploaded)
        suffix = "." + uploaded.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            st.session_state.audio_path = tmp.name

# ------------------ Ghi âm Microphone ------------------
elif audio_source == "Nói trực tiếp (Microphone)":
    duration = st.number_input("⏱ Thời gian ghi âm (giây)", min_value=1, max_value=10, value=3)
    if st.button("🎙️ Ghi âm"):
        st.info("Đang ghi âm...")
        fs = 16000  # sample rate
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            write(tmp.name, fs, recording)
            st.session_state.audio_path = tmp.name
        st.success("✅ Đã ghi âm xong!")
        st.audio(st.session_state.audio_path)

# ------------------ Predict ------------------
if st.button("Predict"):
    if st.session_state.audio_path is None:
        st.warning("⚠️ Vui lòng upload hoặc ghi âm trước")
        st.stop()

    try:
        # Lấy model & hàm
        pack = models_cache[model_name]
        model_kind = pack["kind"]
        model = pack["model"]
        scaler = pack["scaler"]
        get_features = pack["get_features"]
        prepare_input = pack["prepare_input"]

        feats = get_features(st.session_state.audio_path)
        x = prepare_input(feats, scaler)
        label, probs = predict(model_kind, model, x, le)

        st.subheader(f"✅ Kết quả: **{label}**")
        st.write("### Xác suất theo lớp")
        for c, p in zip(le.classes_, probs):
            st.write(f"- {c}: {float(p):.4f}")
        st.write("### Biểu đồ")
        st.bar_chart(probs)

    except Exception as e:
        st.error(f"Lỗi: {e}")

    finally:
        # Xóa file tạm nếu muốn
        try:
            os.remove(st.session_state.audio_path)
        except Exception:
            pass
        st.session_state.audio_path = None
