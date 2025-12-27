import os
import tempfile
import numpy as np
import streamlit as st

from utils.loaders import (
    load_label_encoder,
    load_ml_model,
    load_scaler,
    load_keras_model,
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
    "ML (SVM/RF/KNN...)": {
        "kind": "ml",
        "model_path": os.path.join(ART_DIR, "ml", "model.pkl"),
        "scaler_path": os.path.join(ART_DIR, "ml", "scaler.pkl"),  # nếu không có thì để None
        "get_features": get_features_ml,
        "prepare_input": prepare_input_ml,
    },
    "CNN": {
        "kind": "lstm",
        "model_path": os.path.join(ART_DIR, "lstm", "model.h5"),
        "scaler_path": os.path.join(ART_DIR, "lstm", "scaler.pkl"),  # nếu không có thì để None
        "get_features": get_features_lstm,
        "prepare_input": prepare_input_lstm,
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
        else:
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
    else:
        probs = model.predict(x, verbose=0)[0]

    probs = np.asarray(probs, dtype=np.float32)
    idx = int(np.argmax(probs))
    label = le.inverse_transform([idx])[0]
    return label, probs


st.set_page_config(page_title="SRE/SER Deploy", layout="centered")
st.title("🎙️ SRE/SER Deploy (Upload audio → chọn model → kết quả)")

le, models_cache = load_all()

model_name = st.selectbox("Chọn model", list(MODEL_REGISTRY.keys()))
uploaded = st.file_uploader("Upload audio", type=["wav", "mp3", "flac", "ogg", "m4a"])

if uploaded is not None:
    st.audio(uploaded)

    suffix = "." + uploaded.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        audio_path = tmp.name

    if st.button("Predict"):
        try:
            pack = models_cache[model_name]
            model_kind = pack["kind"]
            model = pack["model"]
            scaler = pack["scaler"]
            get_features = pack["get_features"]
            prepare_input = pack["prepare_input"]

            feats = get_features(audio_path)
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
            try:
                os.remove(audio_path)
            except Exception:
                pass
