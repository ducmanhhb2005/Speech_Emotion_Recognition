import numpy as np
import librosa
import torch
import os
import pickle
# =========================
# CNN-LSTM
# duration=2.5, offset=0.6, sr=22050 (mặc định), hop=512, frame_length=2048
# features = [zcr, rmse, mfcc(flatten)] => 108 + 108 + (20*108) = 2376
# =========================
class CONFIG:


    # --- Tham số trích xuất đặc trưng ---
    SR_FEAT = 16000
    N_MELS = 128
    MAX_PAD_LEN = 200  # ~3.2 giây

    # --- Tham số huấn luyện ---
    BATCH_SIZE = 32
    EPOCHS = 100
    NUM_CLASSES = 7
    INPUT_SHAPE = (1, N_MELS, MAX_PAD_LEN) # (Channels, Height, Width) cho PyTorch

    # --- Mapping (để hiển thị kết quả) ---
    EMOTION_MAP = {
        0: 'neutral', # neutral + calm
        1: 'happy',
        2: 'sad',
        3: 'angry',
        4: 'fearful',
        5: 'disgust',
        6: 'surprised'
    }

    # Corrected: Use EMOTION_MAP directly as it already maps correct IDs to names
    ID_TO_NAME_MAP = EMOTION_MAP
def _zcr(data, frame_length, hop_length):
    z = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(z)


def _rmse(data, frame_length=2048, hop_length=512):
    r = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(r)


def _mfcc(data, sr, hop_length=512, n_mfcc=20, flatten=True):
    m = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    m_t = m.T  # (frames, n_mfcc)
    if flatten:
        return np.ravel(m_t)
    return np.squeeze(m_t)


def get_features_cnn_lstm(path: str):
    """
    Extract features EXACTLY like training pipeline
    Output shape: (162,)
    """
    data, sr = librosa.load(path, duration=2.5, offset=0.6)

    result = np.array([])

    # ZCR (1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    # Chroma STFT (12)
    stft = np.abs(librosa.stft(data))
    chroma = np.mean(
        librosa.feature.chroma_stft(S=stft, sr=sr).T,
        axis=0
    )
    result = np.hstack((result, chroma))

    # MFCC (20)
    mfcc = np.mean(
        librosa.feature.mfcc(y=data, sr=sr).T,
        axis=0
    )
    result = np.hstack((result, mfcc))

    # RMS (1)
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    # MelSpectrogram (128)
    mel = np.mean(
        librosa.feature.melspectrogram(y=data, sr=sr).T,
        axis=0
    )
    result = np.hstack((result, mel))

    return result.astype(np.float32)   # (162,)
def prepare_input_cnn_lstm(feat_1d: np.ndarray, scaler):

    x = feat_1d.reshape(1, -1)   # (1, 162)

    if scaler is not None:
        x = scaler.transform(x)  # scaler fit lúc train

    x = np.expand_dims(x, axis=2)  # (1, 162, 1)
    return x.astype(np.float32)



# =========================
# ML
# =========================
ML_MODEL_PATH = "artifacts/ml/model.pkl"  # svm_model.pkl của m

def get_features_ml(path: str):
    """
    Extract + scale feature cho SVM
    Output: (1, 162)
    """

    # ===== Load audio =====
    y, sr = librosa.load(path, sr=None)
    if y.ndim > 1:
        y = librosa.to_mono(y)

    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    sr = 16000

    if len(y) < sr:
        raise ValueError("Audio quá ngắn")

    # ===== Feature giống lúc train =====
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    rms = librosa.feature.rms(y=y)

    feat = np.hstack([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        delta.mean(axis=1),
        delta2.mean(axis=1),
        rms.mean(),
        rms.std()
    ]).astype(np.float32)

    feat = feat.reshape(1, -1)   # (1, 162)

    # ===== LOAD SCALER TỪ FILE SVM =====
    with open(ML_MODEL_PATH, "rb") as f:
        pack = pickle.load(f)

    scaler = pack["scaler"]
    feat = scaler.transform(feat)

    return feat



def prepare_input_ml(feat_1d: np.ndarray, scaler):
    """
    Chuẩn bị input cho SVM
    """
    x = feat_1d.reshape(1, -1)  # (1, num_features)

    if scaler is not None:
        x = scaler.transform(x)

    return x.astype(np.float32)



# =========================
# CNN
# =========================
def get_features_cnn(path: str):
    """
    Đồng bộ 100% với extract_features trong predict_emotion
    Output: (N_MELS, MAX_PAD_LEN)
    """
    y_audio, _ = librosa.load(path, sr=CONFIG.SR_FEAT)

    if y_audio.ndim > 1:
        y_audio = librosa.to_mono(y_audio)

    melspec = librosa.feature.melspectrogram(
        y=y_audio,
        sr=CONFIG.SR_FEAT,
        n_mels=CONFIG.N_MELS,
        n_fft=2048,
        hop_length=512,
        fmax=8000
    )

    log_melspec = librosa.power_to_db(melspec, ref=np.max)

    # normalize
    log_melspec = (log_melspec - log_melspec.mean()) / (log_melspec.std() + 1e-8)

    # pad / truncate
    if log_melspec.shape[1] < CONFIG.MAX_PAD_LEN:
        pad_width = CONFIG.MAX_PAD_LEN - log_melspec.shape[1]
        log_melspec = np.pad(
            log_melspec,
            pad_width=((0, 0), (0, pad_width)),
            mode="constant"
        )
    else:
        log_melspec = log_melspec[:, :CONFIG.MAX_PAD_LEN]

    return log_melspec.astype(np.float32)




def prepare_input_cnn(feat_2d: np.ndarray, scaler=None):
    """
    Input : (N_MELS, MAX_PAD_LEN)
    Output: (1, 1, N_MELS, MAX_PAD_LEN)
    """
    x = torch.from_numpy(feat_2d).float()
    x = x.unsqueeze(0).unsqueeze(0)
    return x
