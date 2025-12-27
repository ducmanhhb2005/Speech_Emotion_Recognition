import pickle
import joblib
import tensorflow as tf


def load_label_encoder(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_ml_model(path: str):
    return joblib.load(path)


def load_scaler(path: str):
    return joblib.load(path)


def load_keras_model(path: str):
    return tf.keras.models.load_model(path)
