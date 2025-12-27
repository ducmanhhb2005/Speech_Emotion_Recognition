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
    data, sr = librosa.load(path, duration=2.5, offset=0.6)  # sr mặc định 22050
    frame_length = 2048
    hop_length = 512
    n_mfcc = 20

    feat = np.hstack(
        (
            _zcr(data, frame_length, hop_length),
            _rmse(data, frame_length, hop_length),
            _mfcc(data, sr, hop_length=hop_length, n_mfcc=n_mfcc, flatten=True),
        )
    ).astype(np.float32)

    # đảm bảo đúng 2376 (phòng khi file audio khác làm lệch frames)
    target_len = 2376
    if feat.shape[0] < target_len:
        feat = np.pad(feat, (0, target_len - feat.shape[0]))
    else:
        feat = feat[:target_len]

    return feat


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
    TODO: Thay bằng feature ML.
    Gợi ý: MFCC mean/std, chroma, mel, zcr, rmse...
    Output nên là vector 1D cố định (D,)
    """
    data, sr = librosa.load(path, sr=16000, mono=True)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)
    feat = np.concatenate([mean, std], axis=0).astype(np.float32)  # (80,)
    return feat


def prepare_input_ml(feat_1d: np.ndarray, scaler):
    x = feat_1d.reshape(1, -1)
    if scaler is not None:
        x = scaler.transform(x)
    return x.astype(np.float32)


# =========================
# CNN
# =========================
def get_features_cnn(path: str):
    """
    TODO: Thay bằng feature CNN. 
    """
    data, sr = librosa.load(path, sr=16000, mono=True, duration=3.0)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40, hop_length=512)  # (F, frames)

    # pad/trim frames cố định (ví dụ ~94)
    target_frames = int(3.0 * 16000 / 512)
    if mfcc.shape[1] < target_frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, target_frames - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :target_frames]

    # trả (T, F)
    return mfcc.T.astype(np.float32)


def prepare_input_cnn(feat_2d: np.ndarray, scaler):
    x = np.expand_dims(feat_2d, axis=0)
    return x.astype(np.float32)
