import os
import librosa
import soundfile as sf
import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# 1. Cấu hình
# =========================
DATA_DIR = "../input/ravdess-emotional-speech-audio"  # dataset gốc
OUTPUT_DIR = "processed"  # thư mục lưu WAV + CSV
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mapping code -> nhãn text
EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Mapping EmotionID -> label_id theo gốc (0-index)
EMOTION_ID_TO_NUM = {k: int(k)-1 for k in EMOTION_MAP.keys()}
# '01' -> 0, '02' -> 1, ..., '08' -> 7

# =========================
# 2. Hàm preprocess audio
# =========================
def preprocess_audio(file_path, target_sr=16000):
    try:
        y, sr = librosa.load(file_path, sr=None)
        if y.ndim > 1:
            y = librosa.to_mono(y)
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        return y_resampled, target_sr
    except Exception as e:
        print(f"Lỗi khi xử lý {file_path}: {e}")
        return None, None

# =========================
# 3. Duyệt file + lấy nhãn
# =========================
all_files = []
all_labels = []
all_label_ids = []

for root, _, files in os.walk(DATA_DIR):
    for f in files:
        if f.endswith(".wav"):
            path = os.path.join(root, f)
            try:
                code = f.split("-")[2]           # EmotionID từ filename
                label_text = EMOTION_MAP[code]   # nhãn text
                label_id = EMOTION_ID_TO_NUM[code]  # label_id theo gốc
            except:
                print(f"File không chuẩn: {f}")
                continue
            all_files.append(path)
            all_labels.append(label_text)
            all_label_ids.append(label_id)

print(f"Tổng số file: {len(all_files)}")

# =========================
# 4. Chia train/test
# =========================
train_files, test_files, train_labels, test_labels, train_label_ids, test_label_ids = train_test_split(
    all_files, all_labels, all_label_ids,
    test_size=0.2, stratify=all_label_ids, random_state=42
)

# =========================
# 5. Hàm lưu WAV + trả paths
# =========================
def save_wav(files, prefix):
    paths = []
    out_dir = os.path.join(OUTPUT_DIR, prefix)
    os.makedirs(out_dir, exist_ok=True)
    for i, path in enumerate(files):
        y, sr = preprocess_audio(path)
        if y is None:
            continue
        out_name = f"{prefix}_{i}.wav"
        out_path = os.path.join(out_dir, out_name)
        sf.write(out_path, y, sr)
        paths.append(out_path)
    return paths

train_paths = save_wav(train_files, "train")
test_paths = save_wav(test_files, "test")

# =========================
# 6. Lưu CSV final
# =========================
train_df = pd.DataFrame({
    "file_path": train_paths,
    "label": train_labels,
    "label_id": train_label_ids
})

test_df = pd.DataFrame({
    "file_path": test_paths,
    "label": test_labels,
    "label_id": test_label_ids
})

# Chia train -> validation
train_df, val_df = train_test_split(
    train_df, test_size=0.2, stratify=train_df["label_id"], random_state=42
)

# Lưu CSV
train_df.to_csv(os.path.join(OUTPUT_DIR, "train_final.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, "val_final.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test_final.csv"), index=False)

print(f"Số file train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
print("Hoàn tất preprocess, CSV sẵn sàng train mô hình")
print("Bảng label -> label_id:", dict(zip(EMOTION_MAP.values(), EMOTION_ID_TO_NUM.values())))
