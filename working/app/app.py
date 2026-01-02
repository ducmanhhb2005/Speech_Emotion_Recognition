import os
import tempfile
import numpy as np
import streamlit as st
import torch
import sounddevice as sd
from scipy.io.wavfile import write

from utils.loaders import (
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

# ================= CONFIG =================
ART_DIR = "artifacts"

MODEL_REGISTRY = {
    "CNN-LSTM": {
        "kind": "cnn_lstm",
        "model_path": os.path.join(ART_DIR, "cnn_lstm", "model.h5"),
        "scaler_path": os.path.join(ART_DIR, "cnn_lstm", "scaler.pkl"),
        "get_features": get_features_cnn_lstm,
        "prepare_input": prepare_input_cnn_lstm,
    },
    "ML (SVM)": {
        "kind": "ml",
        "model_path": os.path.join(ART_DIR, "ml", "model.pkl"),
        "scaler_path": os.path.join(ART_DIR, "ml", "scaler.pkl"),
        "get_features": get_features_ml,
        "prepare_input": prepare_input_ml,
    },
    "CNN": {
        "kind": "cnn",
        "model_path": os.path.join(ART_DIR, "cnn", "model.pth"),
        "scaler_path": os.path.join(ART_DIR, "cnn", "scaler.pkl"),
        "get_features": get_features_cnn,
        "prepare_input": prepare_input_cnn,
    },
}

EMOTION_MAP = {
    0: 'neutral', 1: 'happy', 2: 'sad',
    3: 'angry', 4: 'fearful', 5: 'disgust', 6: 'surprised'
}

LSTM_MAP = {
    0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy',
    4:'Neutral', 5:'Fear', 6:'Sad', 7:'Surprise'
}

# ================= LOAD MODELS =================
@st.cache_resource
def load_all():
    cache = {}
    for name, cfg in MODEL_REGISTRY.items():
        if cfg["kind"] == "ml":
            model = load_ml_model(cfg["model_path"])
        elif cfg["kind"] == "cnn":
            model = load_torch_model(cfg["model_path"])
        else:
            model = load_keras_model(cfg["model_path"])

        scaler = load_scaler(cfg["scaler_path"]) if os.path.exists(cfg["scaler_path"]) else None

        cache[name] = {**cfg, "model": model, "scaler": scaler}
    return cache

models_cache = load_all()

# ================= PREDICT =================
def predict(kind, model, x):
    if kind == "cnn":
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
        idx = np.argmax(probs)
        return EMOTION_MAP[idx], probs, list(EMOTION_MAP.values())

    if kind == "ml":
        svm = model["svm"] if isinstance(model, dict) else model
        scores = svm.decision_function(x)
        scores = scores[0] if scores.ndim > 1 else scores
        idx = svm.predict(x)[0]
        return EMOTION_MAP[idx], scores, list(EMOTION_MAP.values())

    probs = model.predict(x, verbose=0)[0]
    idx = np.argmax(probs)
    return LSTM_MAP[idx], probs, list(LSTM_MAP.values())

# ================= UI =================
st.set_page_config(page_title="SRE/SER Deploy", layout="centered")
st.title("🎙️ SRE/SER Deploy")

if "audio_path" not in st.session_state:
    st.session_state.audio_path = None

model_name = st.selectbox("Chọn model", list(MODEL_REGISTRY.keys()))
audio_source = st.radio("Chọn nguồn audio", ["Upload file", "Nói trực tiếp (Microphone)"])

# -------- Upload --------
if audio_source == "Upload file":
    uploaded = st.file_uploader("Upload audio", type=["wav", "mp3", "flac", "ogg", "m4a"])
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded.name.split(".")[-1]) as tmp:
            tmp.write(uploaded.read())
            st.session_state.audio_path = tmp.name

# -------- Record --------
if audio_source == "Nói trực tiếp (Microphone)":
    duration = st.number_input("⏱ Thời gian ghi âm (giây)", 1, 10, 3)
    if st.button("🎙️ Ghi âm"):
        st.info("Đang ghi âm...")
        fs = 16000
        rec = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
        sd.wait()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            write(tmp.name, fs, rec)
            st.session_state.audio_path = tmp.name

# ✅ FIX LỖI: LUÔN HIỂN THỊ AUDIO NẾU CÒN FILE
if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
    st.audio(st.session_state.audio_path)

# -------- Predict --------
if st.button("Predict"):
    if not st.session_state.audio_path:
        st.warning("⚠️ Vui lòng upload hoặc ghi âm trước")
        st.stop()

    pack = models_cache[model_name]
    feats = pack["get_features"](st.session_state.audio_path)
    x = pack["prepare_input"](feats, pack["scaler"])

    label, probs, emotions = predict(pack["kind"], pack["model"], x)

    st.subheader(f"✅ Kết quả: **{label}**")
    for e, p in zip(emotions, probs):
        st.write(f"- {e}: {float(p):.4f}")
    st.bar_chart(probs)
