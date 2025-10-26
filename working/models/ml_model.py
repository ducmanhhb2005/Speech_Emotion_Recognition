from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

train_chroma = np.load("working/features/train/chroma.npy", allow_pickle=True)
train_mel = np.load("working/features/train/mel.npy", allow_pickle=True)
train_mfcc = np.load("working/features/train/mfcc.npy", allow_pickle=True)
x_train = np.concatenate((train_chroma,train_mel,train_mfcc), axis=1)

test_chroma = np.load("working/features/test/chroma.npy", allow_pickle=True)
test_mel = np.load("working/features/test/mel.npy", allow_pickle=True)
test_mfcc = np.load("working/features/test/mfcc.npy", allow_pickle=True)
x_test = np.concatenate((test_chroma,test_mel,test_mfcc), axis=1)

x_train = np.mean(x_train, axis=2)
x_test = np.mean(x_test, axis=2)

train_df = pd.read_csv("working/processed/train_final.csv")
y_train = train_df["label_id"].values

test_df = pd.read_csv("working/processed/test_final.csv")
y_test = test_df["label_id"].values

xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(x_train, y_train)
y_pred = xgb_model.predict(x_test)

print(" Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))