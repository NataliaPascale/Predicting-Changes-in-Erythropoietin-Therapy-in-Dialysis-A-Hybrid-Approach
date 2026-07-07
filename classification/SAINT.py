import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, balanced_accuracy_score, f1_score
)
from sklearn.utils.class_weight import compute_class_weight

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)
# =============================================================================
# 1. CONFIG
# =============================================================================
FILE_PATH = #Add the dataset path here

ID_COL = 'id_AnaPaz'
DATE_COL = 'DataReferto'
TARGET_COL = 'DosaggioSettimanaleTotale'

TEST_SIZE_FINAL = 0.20
RANDOM_STATE_SPLIT = 42 


VAL_SPLIT_IN_TRAIN_FINAL = 0.15

# feature engineering
MAX_LAG_DOSE = 2
MAX_LAG_ERI  = 2
DELTA_COLS = ['Emoglobina', 'Ferritina', 'Creatinina', 'Peso Pre', 'T-SAT']


TH_REL = 0.15
TH_UI  = 1000

# training
BATCH_SIZE = 256
EPOCHS_TUNE  = 60 
EPOCHS_FINAL = 120   
PATIENCE = 10
LR_DEFAULT = 3e-4
WEIGHT_DECAY_DEFAULT = 1e-5

# =============================================================================
# 2. LOAD + CLEAN
# =============================================================================
df = pd.read_excel(FILE_PATH)

df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors='coerce')
df = df.dropna(subset=[ID_COL, DATE_COL]).copy()
df = df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)

df = df.dropna(subset=['Emoglobina', 'Peso Pre', TARGET_COL]).copy()

other_features = ['Creatinina', 'Ferritina', 'T-SAT',
                  'Vitamina B12', 'Transferrina', 'ERI']
existing_cols = [c for c in other_features if c in df.columns]
df[existing_cols] = df.groupby(ID_COL)[existing_cols].ffill()

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================
for lag in range(1, MAX_LAG_DOSE + 1):
    df[f'Dose_lag_{lag}'] = df.groupby(ID_COL)[TARGET_COL].shift(lag)

for lag in range(1, MAX_LAG_ERI + 1):
    if 'ERI' in df.columns:
        df[f'ERI_lag_{lag}'] = df.groupby(ID_COL)['ERI'].shift(lag)

for c in DELTA_COLS:
    if c in df.columns:
        df[f'Delta_{c}'] = df.groupby(ID_COL)[c].diff()

df['Hb_std_3'] = (
    df.groupby(ID_COL)['Emoglobina']
      .rolling(3).std()
      .reset_index(level=0, drop=True)
)

def hb_trend(x):
    if len(x) < 3:
        return 0.0
    return np.polyfit(range(len(x)), x, 1)[0]

df['Hb_Trend_5'] = (
    df.groupby(ID_COL)['Emoglobina']
      .transform(lambda x: x.rolling(5, min_periods=3).apply(hb_trend))
)

df['Target_rel'] = (df[TARGET_COL] - df['Dose_lag_1']) / (df['Dose_lag_1'] + 1e-6)
df['Target_rel'] = df['Target_rel'].clip(-0.5, 0.5)

df['EPO_Response'] = (df['Delta_Emoglobina'] / (df['Dose_lag_1'] + 1.0)).clip(-2, 2)
df['Hb_per_EPO']   = (df['Emoglobina'] / (df['Dose_lag_1'] + 1.0)).clip(0, 30)

df['Hb_Volatility'] = (
    df.groupby(ID_COL)['Emoglobina']
      .transform(lambda x: x.rolling(5, min_periods=3).std())
)

if 'Ferritina' in df.columns and 'T-SAT' in df.columns:
    df['Ferritin_TSAT_ratio'] = (df['Ferritina'] / (df['T-SAT'] + 1.0)).clip(0, 200)

df['Dose_var_3'] = (
    df.groupby(ID_COL)[TARGET_COL]
      .rolling(3).std()
      .reset_index(level=0, drop=True)
).clip(0, 10000)

# =============================================================================
# 4. Final Features
# =============================================================================
FEATURE_COLS = [
    'Hb_per_EPO','EPO_Response',
    'Emoglobina','Delta_Emoglobina',
    'Hb_std_3','Hb_Volatility',
    'Dose_lag_1','Dose_lag_2',
    'ERI_lag_1','ERI_lag_2',
    'Dose_var_3',
    'Ferritina','T-SAT','Transferrina','Ferritin_TSAT_ratio',
    'Creatinina','Delta_Creatinina',
    'Peso Pre','Delta_Peso Pre',
    'Vitamina B12',
    'Hb_Trend_5'
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

df_model = df.dropna(subset=['Dose_lag_1', 'Target_rel']).copy()

# =============================================================================
# 5. TARGET CLASS (0=DOWN, 1=KEEP, 2=UP)
# =============================================================================
delta_ui_true = df_model['Dose_lag_1'] * df_model['Target_rel']

is_up   = (df_model['Target_rel'] >= TH_REL) & (delta_ui_true >= TH_UI)
is_down = (df_model['Target_rel'] <= -TH_REL) & (delta_ui_true <= -TH_UI)

df_model['Target_class'] = 1
df_model.loc[is_down, 'Target_class'] = 0
df_model.loc[is_up,   'Target_class'] = 2

print("Rows totali:", len(df_model))
print("Class dist:", df_model['Target_class'].value_counts().sort_index().to_dict())

# =============================================================================
# 6. SPLIT TRAIN/TEST
# =============================================================================
gss_final = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE_FINAL,
    random_state=RANDOM_STATE_SPLIT
)
train_idx, test_idx = next(gss_final.split(df_model, groups=df_model[ID_COL]))

train_all = df_model.iloc[train_idx].copy()
test_final = df_model.iloc[test_idx].copy()

print("\n==================== FINAL SPLIT ====================")
print(f"Patients TRAIN: {train_all[ID_COL].nunique()} | Rows TRAIN: {len(train_all)}")
print(f"Patients TEST : {test_final[ID_COL].nunique()} | Rows TEST : {len(test_final)}")

# =============================================================================
# 7.SAINT MODEL
# =============================================================================
class SAINT_NumOnly(nn.Module):
    def __init__(self, n_features, n_classes=3, d_model=128, n_heads=8, n_layers=4, ff_dim=256, dropout=0.10):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.proj = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_features)])
        self.pos = nn.Parameter(torch.zeros(1, 1 + n_features, d_model))

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True
            ) for _ in range(n_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes)
        )

    def forward(self, x):
        B, F = x.shape
        cls = self.cls.expand(B, -1, -1)
        feats = torch.cat(
            [self.proj[i](x[:, i:i+1]).unsqueeze(1) for i in range(F)],
            dim=1
        )
        z = torch.cat([cls, feats], dim=1) + self.pos[:, :F+1, :]
        for blk in self.blocks:
            z = blk(z)
        return self.head(z[:, 0, :])

# =============================================================================
# 8. UTILS: PREP FOLD (missing ind + imput + scaling FIT)
# =============================================================================
def prep_matrices(train_df, val_df, feature_cols):
   
    isna_cols = [c for c in feature_cols if train_df[c].isna().any()]

    for c in isna_cols:
        train_df[c+"_isna"] = train_df[c].isna().astype(int)
        val_df[c+"_isna"]   = val_df[c].isna().astype(int)

    feature_final = feature_cols + [c+"_isna" for c in isna_cols]

    # imputation
    medians = train_df[feature_cols].median()
    train_df[feature_cols] = train_df[feature_cols].fillna(medians)
    val_df[feature_cols]   = val_df[feature_cols].fillna(medians)

    # scaling
    scaler = StandardScaler()
    train_df[feature_final] = scaler.fit_transform(train_df[feature_final])
    val_df[feature_final]   = scaler.transform(val_df[feature_final])

    X_tr = train_df[feature_final].values.astype(np.float32)
    y_tr = train_df['Target_class'].values.astype(np.int64)
    X_va = val_df[feature_final].values.astype(np.float32)
    y_va = val_df['Target_class'].values.astype(np.int64)

    return X_tr, y_tr, X_va, y_va, len(feature_final), feature_final, medians, scaler, isna_cols

def make_loaders(X_tr, y_tr, X_va, y_va, batch_size):
    train_loader = DataLoader(
        list(zip(torch.tensor(X_tr), torch.tensor(y_tr))),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        list(zip(torch.tensor(X_va), torch.tensor(y_va))),
        batch_size=512, shuffle=False
    )
    return train_loader, val_loader

def train_earlystop(model, train_loader, val_loader, ce, opt, epochs, patience):
    best_val = 1e18
    best_state = None
    wait = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = ce(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        vloss = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                vloss += ce(model(xb), yb).item()
                n += 1
        vloss = vloss / max(1, n)

        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def eval_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    danger = cm[0,2] + cm[2,0]
    return acc, bacc, mf1, danger, cm

# =============================================================================
# 9. CV TRAIN: GRID SEARCH HYPERPARAMETERS
# =============================================================================

GRID = {
    "d_model":   [96, 128],
    "n_layers":  [3, 4],
    "dropout":   [0.10, 0.15],
    "lr":        [2e-4, 3e-4],
    "weight_decay": [1e-5]
}

def valid_heads(d_model):
    if d_model % 8 == 0:
        return 8
    if d_model % 4 == 0:
        return 4
    return 2

gkf = GroupKFold(n_splits=5) 

param_list = list(itertools.product(
    GRID["d_model"], GRID["n_layers"], GRID["dropout"], GRID["lr"], GRID["weight_decay"]
))

results = []
print("\n==================== CV TRAIN (SAINT) ====================")

for i, (d_model, n_layers, dropout, lr, wd) in enumerate(param_list, 1):
    n_heads = valid_heads(d_model)

    fold_scores = []
    fold_baccs = []
    fold_dangers = []

    print(f"\nConfig {i}/{len(param_list)} | d_model={d_model}, heads={n_heads}, layers={n_layers}, drop={dropout}, lr={lr}, wd={wd}")

    for fold, (tr_i, va_i) in enumerate(gkf.split(train_all, groups=train_all[ID_COL]), 1):
        tr_df = train_all.iloc[tr_i].copy()
        va_df = train_all.iloc[va_i].copy()

        X_tr, y_tr, X_va, y_va, n_feat, _, _, _, _ = prep_matrices(tr_df, va_df, FEATURE_COLS)

        train_loader, val_loader = make_loaders(X_tr, y_tr, X_va, y_va, BATCH_SIZE)

        # class weights dal train del fold
        cw = compute_class_weight(class_weight="balanced", classes=np.array([0,1,2]), y=y_tr)
        ce = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(DEVICE))

        model = SAINT_NumOnly(
            n_features=n_feat, n_classes=3,
            d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, ff_dim=256,
            dropout=dropout
        ).to(DEVICE)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        model = train_earlystop(model, train_loader, val_loader, ce, opt, EPOCHS_TUNE, PATIENCE)

        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(torch.tensor(X_va).to(DEVICE)), dim=1).cpu().numpy()

        acc, bacc, mf1, danger, _ = eval_metrics(y_va, preds)

        fold_scores.append(mf1)
        fold_baccs.append(bacc)
        fold_dangers.append(danger)

        print(f"  Fold {fold}/5 | Macro-F1={mf1:.4f} | BalAcc={bacc:.4f} | Danger={danger}")

    mean_mf1 = float(np.mean(fold_scores))
    mean_bacc = float(np.mean(fold_baccs))
    mean_danger = float(np.mean(fold_dangers))

    results.append({
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": wd,
        "cv_macro_f1": mean_mf1,
        "cv_bal_acc": mean_bacc,
        "cv_danger": mean_danger
    })


res_df = pd.DataFrame(results).sort_values(
    by=["cv_macro_f1", "cv_bal_acc", "cv_danger"],
    ascending=[False, False, True]
).reset_index(drop=True)

print("\n==================== BEST ====================")
print(res_df.head(10))

best = res_df.iloc[0].to_dict()
print("\nBest params:", best)

# =============================================================================
# 10. FINAL TRAIN AND TEST
# =============================================================================

gss_val = GroupShuffleSplit(
    n_splits=1,
    test_size=VAL_SPLIT_IN_TRAIN_FINAL,
    random_state=RANDOM_STATE_SPLIT
)
tr_i, va_i = next(gss_val.split(train_all, groups=train_all[ID_COL]))

train_df_final = train_all.iloc[tr_i].copy()
val_df_final   = train_all.iloc[va_i].copy()
test_df_final  = test_final.copy()

# Prep: missing ind + imput + scaling FIT
def prep_train_val_test(train_df, val_df, test_df, feature_cols):
    isna_cols = [c for c in feature_cols if train_df[c].isna().any()]

    for c in isna_cols:
        train_df[c+"_isna"] = train_df[c].isna().astype(int)
        val_df[c+"_isna"]   = val_df[c].isna().astype(int)
        test_df[c+"_isna"]  = test_df[c].isna().astype(int)

    feature_final = feature_cols + [c+"_isna" for c in isna_cols]

    medians = train_df[feature_cols].median()
    train_df[feature_cols] = train_df[feature_cols].fillna(medians)
    val_df[feature_cols]   = val_df[feature_cols].fillna(medians)
    test_df[feature_cols]  = test_df[feature_cols].fillna(medians)

    scaler = StandardScaler()
    train_df[feature_final] = scaler.fit_transform(train_df[feature_final])
    val_df[feature_final]   = scaler.transform(val_df[feature_final])
    test_df[feature_final]  = scaler.transform(test_df[feature_final])

    X_tr = train_df[feature_final].values.astype(np.float32)
    y_tr = train_df['Target_class'].values.astype(np.int64)
    X_va = val_df[feature_final].values.astype(np.float32)
    y_va = val_df['Target_class'].values.astype(np.int64)
    X_te = test_df[feature_final].values.astype(np.float32)
    y_te = test_df['Target_class'].values.astype(np.int64)

    return X_tr, y_tr, X_va, y_va, X_te, y_te, len(feature_final), feature_final, medians, scaler, isna_cols

X_tr, y_tr, X_va, y_va, X_te, y_te, n_feat, feature_final, medians, scaler, isna_cols = prep_train_val_test(
    train_df_final, val_df_final, test_df_final, FEATURE_COLS
)


print(f"Patients TRAIN_final: {train_df_final[ID_COL].nunique()} | Rows: {len(train_df_final)}")
print(f"Patients VAL_final  : {val_df_final[ID_COL].nunique()} | Rows: {len(val_df_final)}")
print(f"Patients TEST_final : {test_df_final[ID_COL].nunique()} | Rows: {len(test_df_final)}")

train_loader, val_loader = make_loaders(X_tr, y_tr, X_va, y_va, BATCH_SIZE)

cw = compute_class_weight(class_weight="balanced", classes=np.array([0,1,2]), y=y_tr)
ce = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(DEVICE))

model_final = SAINT_NumOnly(
    n_features=n_feat, n_classes=3,
    d_model=int(best["d_model"]),
    n_heads=int(best["n_heads"]),
    n_layers=int(best["n_layers"]),
    ff_dim=256,
    dropout=float(best["dropout"])
).to(DEVICE)

opt = torch.optim.AdamW(
    model_final.parameters(),
    lr=float(best["lr"]),
    weight_decay=float(best["weight_decay"])
)

model_final = train_earlystop(model_final, train_loader, val_loader, ce, opt, EPOCHS_FINAL, PATIENCE)

# Eval su TEST finale
model_final.eval()
with torch.no_grad():
    preds_test = torch.argmax(model_final(torch.tensor(X_te).to(DEVICE)), dim=1).cpu().numpy()

acc, bacc, mf1, danger, cm = eval_metrics(y_te, preds_test)

print("\n=== TEST  ===")
print("Accuracy:", acc)
print("Balanced Acc:", bacc)
print("Macro-F1:", mf1)
print("Danger DOWN<->UP:", danger)
print("\nConfusion matrix (rows=true cols=pred):\n", cm)
print("\nClassification report:\n", classification_report(y_te, preds_test))

import os, torch, joblib

SAVE_PATH = "/content/drive/MyDrive/Colab Notebooks/SAINT_EPO_model_ACC0615"
os.makedirs(SAVE_PATH, exist_ok=True)

torch.save(model_final.state_dict(), f"{SAVE_PATH}/model_weights.pt")
joblib.dump(scaler, f"{SAVE_PATH}/scaler.pkl")

metadata = {
    "feature_cols": FEATURE_COLS,
    "feature_final": feature_final,
    "isna_cols": isna_cols,
    "medians": medians,
    "best_params": best,   
    "n_feat": n_feat,
    "test_acc": float(acc),         
    "test_macro_f1": float(mf1),     
    "test_bal_acc": float(bacc),     
    "test_danger": int(danger)       
}
joblib.dump(metadata, f"{SAVE_PATH}/metadata.pkl")

print("AINT (ACC~0.615) saved in:", SAVE_PATH)
print("Best params saved:", best)

meta_check = joblib.load(f"{SAVE_PATH}/metadata.pkl")

