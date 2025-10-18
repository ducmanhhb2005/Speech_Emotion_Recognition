import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# =========================
# 1. Cấu hình
# =========================
SAMPLE_RATE = 16000     # phải khớp với preprocess_audio
N_MELS = 128
N_MFCC = 13
HOP_LENGTH = 512
N_FFT = 2048
DURATION = 3.0           # cắt/pad độ dài cố định

DATA_DIR = "processed"
OUTPUT_DIR = "features"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_audio_fixed_length(file_path, sr=SAMPLE_RATE, duration=DURATION):
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    else:
        y = y[:target_len]
    return y


def extract_features(file_path):
    y = load_audio_fixed_length(file_path)
    sr = SAMPLE_RATE

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT,
                                         hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    return mel_db, chroma, mfcc

#process csv file
def process_csv(csv_path, split_name):
    print(f"\n🔹 Đang xử lý: {split_name}")
    df = pd.read_csv(csv_path)
    out_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(out_dir, exist_ok=True)

    all_mel, all_chroma, all_mfcc, all_labels = [], [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        fpath = row["file_path"]
        label_id = row["label_id"]
        try:
            mel, chroma, mfcc = extract_features(fpath)
            all_mel.append(mel)
            all_chroma.append(chroma)
            all_mfcc.append(mfcc)
            all_labels.append(label_id)
        except Exception as e:
            print(f"Error {fpath}: {e}")

    # Lưu thành numpy
    np.save(os.path.join(out_dir, "mel.npy"), np.array(all_mel, dtype=object))
    np.save(os.path.join(out_dir, "chroma.npy"), np.array(all_chroma, dtype=object))
    np.save(os.path.join(out_dir, "mfcc.npy"), np.array(all_mfcc, dtype=object))
    np.save(os.path.join(out_dir, "labels.npy"), np.array(all_labels))

    print(f" Saved {split_name}: {len(all_labels)} example")


if __name__ == "__main__":
    process_csv(os.path.join(DATA_DIR, "train_final.csv"), "train")
    process_csv(os.path.join(DATA_DIR, "val_final.csv"), "val")
    process_csv(os.path.join(DATA_DIR, "test_final.csv"), "test")

    print("\n Done")
