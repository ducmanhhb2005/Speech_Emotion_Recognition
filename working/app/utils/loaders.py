import pickle
import joblib
import tensorflow as tf
import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        # Global Pooling and Dense layers
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = self.dense_block(x)
        return x
    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            probs = F.softmax(outputs, dim=1)  # Chuyển sang xác suất
        return probs
def load_label_encoder(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_ml_model(path: str):
    return joblib.load(path)


def load_scaler(path: str):
    return joblib.load(path)


def load_keras_model(path: str):
    return tf.keras.models.load_model(path)

def load_torch_model(model_path):
    model = SimpleCNN()
    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )
    model.eval()
    return model