import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from tqdm import tqdm
import joblib
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import StackingRegressor

# Load Wav2Vec2
bundle = torchaudio.pipelines.WAV2VEC2_BASE
wav2vec_model = bundle.get_model()
wav2vec_model.eval()

def extract_wav2vec_features(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != bundle.sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=bundle.sample_rate)
        waveform = resampler(waveform)
    with torch.inference_mode():
        features = wav2vec_model(waveform)[0].squeeze(0)
        mean_feat = features.mean(dim=0)
        std_feat = features.std(dim=0)
        duration = waveform.shape[1] / sample_rate
        final_features = torch.cat([mean_feat, std_feat, torch.tensor([duration])])
        return final_features.numpy()

# Load training data
train_df = pd.read_csv("/kaggle/input/shl-intern-hiring-assessment/dataset/train.csv")

# Load or extract features
if os.path.exists("X_wav2vec.npy") and os.path.exists("y_wav2vec.npy"):
    print("🔁 Loading saved features...")
    X = np.load("X_wav2vec.npy")
    y = np.load("y_wav2vec.npy")
else:
    print("🎧 Extracting features...")
    X, y = [], []
    for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
        try:
            path = f"/kaggle/input/shl-intern-hiring-assessment/dataset/audios_train/{row['filename']}"
            feat = extract_wav2vec_features(path)
            X.append(feat)
            y.append(row['label'])
        except Exception as e:
            print(f"❌ Error with {row['filename']}: {e}")

    target_shape = X[0].shape
    X_filtered, y_filtered = [], []
    for i, feat in enumerate(X):
        if isinstance(feat, np.ndarray) and feat.shape == target_shape:
            X_filtered.append(feat)
            y_filtered.append(y[i])

    X = np.array(X_filtered)
    y = np.array(y_filtered)
    np.save("X_wav2vec.npy", X)
    np.save("y_wav2vec.npy", y)

# Scale
scaler_path = "scaler.pkl"
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    scaler = StandardScaler()
    scaler.fit(X)
    joblib.dump(scaler, scaler_path)
X = scaler.transform(X)

# Define base models
xgb_reg = xgb.XGBRegressor(n_estimators=250, learning_rate=0.05, max_depth=6, subsample=0.8, random_state=42)
lgb_reg = lgb.LGBMRegressor(n_estimators=250, learning_rate=0.05, max_depth=6, subsample=0.8, random_state=42)
cb_reg = cb.CatBoostRegressor(iterations=250, learning_rate=0.05, depth=6, verbose=0, random_state=42)

# K-Fold for stacking
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((X.shape[0], 3))

for i, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {i+1}/5")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    xgb_reg.fit(X_train, y_train)
    lgb_reg.fit(X_train, y_train)
    cb_reg.fit(X_train, y_train)

    oof_preds[val_idx, 0] = xgb_reg.predict(X_val)
    oof_preds[val_idx, 1] = lgb_reg.predict(X_val)
    oof_preds[val_idx, 2] = cb_reg.predict(X_val)

meta_model = Ridge()
meta_model.fit(oof_preds, y)
joblib.dump(meta_model, "meta_model.pkl")
joblib.dump(xgb_reg, "xgb_model.pkl")
joblib.dump(lgb_reg, "lgb_model.pkl")
joblib.dump(cb_reg, "cb_model.pkl")

# Validation score
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
val_preds = meta_model.predict(np.vstack([
    xgb_reg.predict(X_val),
    lgb_reg.predict(X_val),
    cb_reg.predict(X_val)
]).T)
corr, _ = pearsonr(y_val, val_preds)
print("📈 Pearson Correlation on Validation:", corr)

# Prediction Function
def stacked_predict(features):
    xgb_pred = xgb_reg.predict(features)
    lgb_pred = lgb_reg.predict(features)
    cb_pred = cb_reg.predict(features)
    stacked_input = np.vstack([xgb_pred, lgb_pred, cb_pred]).T
    return meta_model.predict(stacked_input)

# Predict test
test_df = pd.read_csv("/kaggle/input/shl-intern-hiring-assessment/dataset/test.csv")
submission = []
print("🧪 Predicting test data...")
for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
    try:
        path = f"/kaggle/input/shl-intern-hiring-assessment/dataset/audios_test/{row['filename']}"
        features = extract_wav2vec_features(path)
        features = scaler.transform([features])
        pred = stacked_predict(features)[0]
        submission.append([row['filename'], pred])
    except Exception as e:
        print(f"❌ Error processing {row['filename']}: {e}")
        submission.append([row['filename'], 0.0])

submission_df = pd.DataFrame(submission, columns=['filename', 'label'])
submission_df.to_csv("submission.csv", index=False)
print("✅ Submission saved as submission.csv")
