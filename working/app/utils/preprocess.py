import numpy as np
import librosa


# =========================
# CNN-LSTM
# duration=2.5, offset=0.6, sr=22050 (mặc định), hop=512, frame_length=2048
# features = [zcr, rmse, mfcc(flatten)] => 108 + 108 + (20*108) = 2376
# =========================
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
    Extract MFCC features exactly like training pipeline
    Output shape: (30,)
    """
    data, sr = librosa.load(path, duration=2.5, offset=0.6)

    # MFCC giống hệt lúc train
    mfccs = librosa.feature.mfcc(
        y=data,
        sr=sr,
        n_mfcc=30
    )

    # Mean theo time axis
    mfccs_mean = np.mean(mfccs.T, axis=0)  # (30,)

    return mfccs_mean.astype(np.float32)


def prepare_input_cnn_lstm(feat_1d: np.ndarray, scaler):
    x = feat_1d.reshape(1, -1)  # (1, 2376)
    if scaler is not None:
        x = scaler.transform(x)
    x = np.expand_dims(x, axis=2)  # (1, 2376, 1)
    return x.astype(np.float32)


# =========================
# ML
# =========================
def get_features_ml(path: str):
    """
    MUST match extract_feature_from_raw used in training
    Output shape: (180,)
    """
    y, sr = librosa.load(path, sr=None, mono=True)

    result = np.array([])

    # --- MFCC (40) ---
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs, axis=1)
    result = np.hstack((result, mfccs))

    # --- CHROMA (12) ---
    stft = np.abs(librosa.stft(y))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    chroma = np.mean(chroma, axis=1)
    result = np.hstack((result, chroma))

    # --- MEL (128) ---
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel = np.mean(mel, axis=1)
    result = np.hstack((result, mel))

    return result.astype(np.float32)


def prepare_input_ml(feat_1d: np.ndarray, scaler):
    x = feat_1d.reshape(1, -1)
    if scaler is not None:
        x = scaler.transform(x)
    return x.astype(np.float32)


# =========================
# CNN
# =========================
def get_features_cnn(path: str, sr=16000, n_mels=128, max_pad_len=200):
    data, _ = librosa.load(path, sr=sr, mono=True, duration=3.2)
    melspec = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512)
    log_melspec = librosa.power_to_db(melspec, ref=np.max)
    log_melspec = (log_melspec - log_melspec.mean()) / (log_melspec.std() + 1e-8)

    # Pad hoặc truncate
    if log_melspec.shape[1] < max_pad_len:
        pad_width = max_pad_len - log_melspec.shape[1]
        log_melspec = np.pad(log_melspec, ((0,0),(0,pad_width)), mode='constant')
    else:
        log_melspec = log_melspec[:, :max_pad_len]

    return log_melspec.T.astype(np.float32)  # shape (max_pad_len, n_mels)



def prepare_input_cnn(feat_2d: np.ndarray, scaler=None):
    """
    Chuyển MFCC (T, F) thành input CNN dạng (1, T, F) float32.
    Có thể dùng scaler để chuẩn hóa nếu cần.
    """
    x = np.expand_dims(feat_2d, axis=0)  # (1, T, F)
    if scaler is not None:
        T, F = x.shape[1], x.shape[2]
        x_reshaped = x.reshape(-1, F)
        x_scaled = scaler.transform(x_reshaped)
        x = x_scaled.reshape(1, T, F)
    return x.astype(np.float32)