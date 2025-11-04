# extract_features_advanced.py

import os
import numpy as np
import pandas as pd
import librosa
from scipy.stats import skew, kurtosis
from tqdm import tqdm

# =========================
# 1. Cấu hình (Giữ nguyên như file cũ)
# =========================
SAMPLE_RATE = 16000
N_MELS = 128
N_MFCC = 40  # Tăng số lượng MFCC để có thêm thông tin
HOP_LENGTH = 512
N_FFT = 2048
DURATION = 3.0

DATA_DIR = "../processed"  # Sử dụng thư mục đã xử lý từ bước trước
OUTPUT_DIR = "../features_advanced" # Lưu vào thư mục mới để không bị lẫn
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_audio_fixed_length(file_path, sr=SAMPLE_RATE, duration=DURATION):
    """Tải audio và pad/cắt về độ dài cố định."""
    try:
        y, _ = librosa.load(file_path, sr=sr, mono=True)
        target_len = int(sr * duration)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode="constant")
        else:
            y = y[:target_len]
        return y
    except Exception as e:
        print(f"Lỗi khi tải file {file_path}: {e}")
        return None

def get_feature_stats(features):
    """
    Tính toán các đặc trưng thống kê từ một ma trận đặc trưng (ví dụ: MFCC).
    Input: features (shape: [n_features, time_steps])
    Output: một vector 1D chứa các giá trị thống kê.
    """
    result = []
    # Mean, Std, Skew, Kurtosis, Median, Min, Max
    result.extend(np.mean(features, axis=1))
    result.extend(np.std(features, axis=1))
    result.extend(skew(features, axis=1))
    result.extend(kurtosis(features, axis=1))
    result.extend(np.median(features, axis=1))
    result.extend(np.min(features, axis=1))
    result.extend(np.max(features, axis=1))
    return result


def extract_advanced_features(file_path):
    """
    Trích xuất một bộ đặc trưng âm học đầy đủ và trả về một vector 1D.
    """
    y = load_audio_fixed_length(file_path)
    if y is None:
        return None

    sr = SAMPLE_RATE
    feature_vector = []

    # 1. MFCCs và các delta
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    feature_vector.extend(get_feature_stats(mfcc))
    feature_vector.extend(get_feature_stats(delta_mfcc))
    feature_vector.extend(get_feature_stats(delta2_mfcc))
    
    # 2. Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    feature_vector.extend(get_feature_stats(mel_db))

    # 3. Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    feature_vector.extend(get_feature_stats(chroma))

    # 4. Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    feature_vector.extend(get_feature_stats(contrast))

    # 5. Zero Crossing Rate & RMS Energy (Đây là các vector 1D, cần reshape)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    feature_vector.extend(get_feature_stats(zcr.reshape(1, -1))) # reshape thành (1, time)
    feature_vector.extend(get_feature_stats(rms.reshape(1, -1)))
    
    return np.array(feature_vector, dtype=np.float32)

def process_csv(csv_path, split_name):
    """
    Đọc file CSV, trích xuất đặc trưng cho từng file và lưu thành file .npy
    """
    print(f"\n🔹 Đang xử lý tập: {split_name}")
    df = pd.read_csv(csv_path)
    out_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(out_dir, exist_ok=True)

    all_features, all_labels = [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        fpath = os.path.join("..", row["file_path"])
        label_id = row["label_id"]
        
        features = extract_advanced_features(fpath)
        if features is not None:
            all_features.append(features)
            all_labels.append(label_id)

    # Lưu thành 2 file: một cho features, một cho labels
    np.save(os.path.join(out_dir, "features.npy"), np.array(all_features))
    np.save(os.path.join(out_dir, "labels.npy"), np.array(all_labels))

    print(f" Đã lưu {split_name}: {len(all_labels)} mẫu.")
    print(f"   Kích thước vector đặc trưng của mỗi mẫu: {all_features[0].shape[0]}")


if __name__ == "__main__":
    process_csv(os.path.join(DATA_DIR, "train_final.csv"), "train")
    process_csv(os.path.join(DATA_DIR, "val_final.csv"), "val")
    process_csv(os.path.join(DATA_DIR, "test_final.csv"), "test")

    print("\n Hoàn tất trích xuất đặc trưng nâng cao!")