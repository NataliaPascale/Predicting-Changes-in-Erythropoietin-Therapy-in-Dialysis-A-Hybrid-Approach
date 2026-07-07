import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import xgboost as xgb

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

# =============================================================================
# 1) PATH SAINT
# =============================================================================
SAVE_PATH = #Add the SAINT model path here
WEIGHTS_PT = f"{SAVE_PATH}/model_weights.pt"
SCALER_PKL = f"{SAVE_PATH}/scaler.pkl"
META_PKL   = f"{SAVE_PATH}/metadata.pkl"

assert os.path.exists(WEIGHTS_PT), "I can't find model_weights.pt"
assert os.path.exists(SCALER_PKL), "I can't find scaler.pkl"
assert os.path.exists(META_PKL),   "I can't find metadata.pkl"

meta   = joblib.load(META_PKL)
scaler = joblib.load(SCALER_PKL)

SAINT_FEATURE_COLS = meta["feature_cols"]      
feature_final      = meta["feature_final"]    
isna_cols          = meta["isna_cols"]        
medians            = meta["medians"]          
best_saint         = meta["best_params"]      
n_feat             = meta["n_feat"]           

print("Metadata SAINT loaded. n_feat:", n_feat)
if "test_acc" in meta:
    print("SAINT saved test_acc:", meta["test_acc"])

# =============================================================================
# 2) CONFIG DATASET + SPLIT
# =============================================================================
FILE_PATH = #Add the dataset path here
ID_COL = "id_AnaPaz"
DATE_COL = "DataReferto"
TARGET_COL = "DosaggioSettimanaleTotale"

TEST_SIZE = 0.20
RANDOM_STATE = 42
VAL_SPLIT_IN_TRAIN = 0.15

TH_REL = 0.15
TH_UI  = 1000

# XGB best params (i tuoi)
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.05,
    "max_depth": 4,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "alpha": 0.5,
    **{"lambda": 2.0}, 
}

NUM_BOOST_ROUND = 800
EARLY_STOP = 50

# =============================================================================
# 3) SAINT MODEL
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
        feats = torch.cat([self.proj[i](x[:, i:i+1]).unsqueeze(1) for i in range(F)], dim=1)
        z = torch.cat([cls, feats], dim=1) + self.pos[:, :F+1, :]
        for blk in self.blocks:
            z = blk(z)
        return self.head(z[:, 0, :])

model_saint = SAINT_NumOnly(
    n_features=n_feat,
    n_classes=3,
    d_model=int(best_saint["d_model"]),
    n_heads=int(best_saint["n_heads"]),
    n_layers=int(best_saint["n_layers"]),
    ff_dim=256,
    dropout=float(best_saint["dropout"])
).to(DEVICE)

model_saint.load_state_dict(torch.load(WEIGHTS_PT, map_location="cpu"))
model_saint.eval()
print("SAINT loaded.")

# =============================================================================
# 4) LOAD + CLEAN + FEATURE ENGINEERING
# =============================================================================
df = pd.read_excel(FILE_PATH)

df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
df = df.dropna(subset=[ID_COL, DATE_COL]).copy()
df = df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)

# min required
df = df.dropna(subset=["Emoglobina", "Peso Pre", TARGET_COL]).copy()

# ffill alcune colonne (come tuoi script)
fill_cols = ["Creatinina", "Ferritina", "T-SAT", "Vitamina B12", "Transferrina", "ERI"]
fill_cols = [c for c in fill_cols if c in df.columns]
df[fill_cols] = df.groupby(ID_COL)[fill_cols].ffill()

# lag dose
df["Dose_lag_1"] = df.groupby(ID_COL)[TARGET_COL].shift(1)
df["Dose_lag_2"] = df.groupby(ID_COL)[TARGET_COL].shift(2)

# lag ERI
if "ERI" in df.columns:
    df["ERI_lag_1"] = df.groupby(ID_COL)["ERI"].shift(1)
    df["ERI_lag_2"] = df.groupby(ID_COL)["ERI"].shift(2)

# delta
for c in ["Emoglobina", "Ferritina", "Creatinina", "Peso Pre", "T-SAT"]:
    if c in df.columns:
        df[f"Delta_{c}"] = df.groupby(ID_COL)[c].diff()

# Hb_std_3
df["Hb_std_3"] = (
    df.groupby(ID_COL)["Emoglobina"]
      .rolling(3).std()
      .reset_index(level=0, drop=True)
)

# Hb trend 5
def hb_trend(x):
    if len(x) < 3:
        return 0.0
    return np.polyfit(range(len(x)), x, 1)[0]

df["Hb_Trend_5"] = (
    df.groupby(ID_COL)["Emoglobina"]
      .transform(lambda x: x.rolling(5, min_periods=3).apply(hb_trend))
)

df["Hb_Volatility"] = (
    df.groupby(ID_COL)["Emoglobina"]
      .transform(lambda x: x.rolling(5, min_periods=3).std())
)

# target rel
df["Target_rel"] = (df[TARGET_COL] - df["Dose_lag_1"]) / (df["Dose_lag_1"] + 1e-6)
df["Target_rel"] = df["Target_rel"].clip(-0.5, 0.5)


df["EPO_Response"] = (df["Delta_Emoglobina"] / (df["Dose_lag_1"] + 1.0)).clip(-2, 2)
df["Hb_per_EPO"]   = (df["Emoglobina"] / (df["Dose_lag_1"] + 1.0)).clip(0, 30)

if "Ferritina" in df.columns and "T-SAT" in df.columns:
    df["Ferritin_TSAT_ratio"] = (df["Ferritina"] / (df["T-SAT"] + 1.0)).clip(0, 200)

df["Dose_var_3"] = (
    df.groupby(ID_COL)[TARGET_COL]
      .rolling(3).std()
      .reset_index(level=0, drop=True)
).clip(0, 10000)


df_model = df.dropna(subset=["Dose_lag_1", "Target_rel"]).copy()


delta_ui_true = df_model["Dose_lag_1"] * df_model["Target_rel"]
is_up   = (df_model["Target_rel"] >= TH_REL)  & (delta_ui_true >= TH_UI)
is_down = (df_model["Target_rel"] <= -TH_REL) & (delta_ui_true <= -TH_UI)

df_model["Target_class"] = 1
df_model.loc[is_down, "Target_class"] = 0
df_model.loc[is_up,   "Target_class"] = 2

print("Rows totali:", len(df_model))
print("Class dist :", df_model["Target_class"].value_counts().sort_index().to_dict())

# =============================================================================
# 5) SPLIT TRAIN/TEST 
# =============================================================================
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
tr_idx, te_idx = next(gss.split(df_model, groups=df_model[ID_COL]))

train_all = df_model.iloc[tr_idx].copy()
test_all  = df_model.iloc[te_idx].copy()

print("\n==================== FINAL SPLIT ====================")
print("Pazienti TRAIN:", train_all[ID_COL].nunique(), "| Righe TRAIN:", len(train_all))
print("Pazienti TEST :", test_all[ID_COL].nunique(),  "| Righe TEST :", len(test_all))

# =============================================================================
# 6) PREPROCESS FOR SAINT 
# =============================================================================
def preprocess_for_saint(df_in: pd.DataFrame) -> np.ndarray:
    dfp = df_in.copy()

    
    for c in isna_cols:
        dfp[c + "_isna"] = dfp[c].isna().astype(int)

    
    dfp[SAINT_FEATURE_COLS] = dfp[SAINT_FEATURE_COLS].fillna(medians)

    
    X_df = dfp[feature_final]
    X = scaler.transform(X_df).astype(np.float32)
    return X

def saint_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor(X).to(DEVICE)), dim=1).cpu().numpy()
    return probs

X_saint_tr = preprocess_for_saint(train_all)
X_saint_te = preprocess_for_saint(test_all)

probs_tr = saint_proba(model_saint, X_saint_tr)
probs_te = saint_proba(model_saint, X_saint_te)  

# =============================================================================
# 7) FEATURE FOR XGB
# =============================================================================
XGB_FEATURE_COLS = [
    "Hb_per_EPO", "EPO_Response",
    "Emoglobina", "Delta_Emoglobina",
    "Hb_std_3", "Hb_Trend_5",
    "Dose_lag_1", "Dose_lag_2",
    "ERI_lag_1", "ERI_lag_2",
    "Ferritina", "T-SAT", "Transferrina",
    "Ferritin_TSAT_ratio",
    "Creatinina", "Delta_Creatinina",
    "Peso Pre", "Delta_Peso Pre",
    "Vitamina B12",
    "Dose_var_3"
]
XGB_FEATURE_COLS = [c for c in XGB_FEATURE_COLS if c in df_model.columns]

# =============================================================================
# 8) Function XGB: train with val interna + predict
# =============================================================================
def fit_xgb_with_internal_val(train_df_full: pd.DataFrame, params: dict):
    gss_val = GroupShuffleSplit(n_splits=1, test_size=VAL_SPLIT_IN_TRAIN, random_state=RANDOM_STATE)
    tr_i, va_i = next(gss_val.split(train_df_full, groups=train_df_full[ID_COL]))

    tr_df = train_df_full.iloc[tr_i].copy()
    va_df = train_df_full.iloc[va_i].copy()

    scaler_y = StandardScaler()

    X_tr = tr_df[XGB_FEATURE_COLS].values
    X_va = va_df[XGB_FEATURE_COLS].values

    y_tr = tr_df[["Target_rel"]].values
    y_va = va_df[["Target_rel"]].values

    y_tr_s = scaler_y.fit_transform(y_tr).reshape(-1)
    y_va_s = scaler_y.transform(y_va).reshape(-1)

    dtrain = xgb.DMatrix(X_tr, label=y_tr_s, missing=np.nan)
    dval   = xgb.DMatrix(X_va, label=y_va_s, missing=np.nan)

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=EARLY_STOP,
        verbose_eval=False
    )
    return booster, scaler_y

def predict_rel(booster, scaler_y, df_ref: pd.DataFrame) -> np.ndarray:
    X = df_ref[XGB_FEATURE_COLS].values
    dmat = xgb.DMatrix(X, missing=np.nan)
    pred_s = booster.predict(dmat).reshape(-1, 1)
    pred_rel = scaler_y.inverse_transform(pred_s).reshape(-1)
    pred_rel = np.clip(pred_rel, -0.5, 0.5)
    return pred_rel.astype(np.float32)

def eval_ui(df_ref: pd.DataFrame, pred_rel: np.ndarray):
    true_rel = df_ref["Target_rel"].values.astype(np.float32)
    dose_prev = df_ref["Dose_lag_1"].values.astype(np.float32)

    true_dose = np.maximum(0.0, dose_prev * (1.0 + true_rel))
    pred_dose = np.maximum(0.0, dose_prev * (1.0 + pred_rel))

   
    mae = mean_absolute_error(true_dose, pred_dose)
    rmse = np.sqrt(mean_squared_error(true_dose, pred_dose))
    r2 = r2_score(true_dose, pred_dose)
    within2k = float(np.mean(np.abs(pred_dose - true_dose) <= 2000.0))


    pred_round = np.round(pred_dose / 1000.0) * 1000.0
    acc_round = float(np.mean(np.abs(pred_round - true_dose) <= 2000.0))

    return mae, rmse, r2, within2k, acc_round


# =============================================================================
# 9) BASELINE XGB 
# =============================================================================
base_booster, base_scaler_y = fit_xgb_with_internal_val(train_all, XGB_PARAMS)
pred_rel_base = predict_rel(base_booster, base_scaler_y, test_all)
mae_b, rmse_b, r2_b, w2k_b, acc_round_b = eval_ui(test_all, pred_rel_base)

print("\n==================== BASELINE XGB (BEST) ====================")
print(f"MAE UI : {mae_b:.0f}")
print(f"RMSE UI: {rmse_b:.0f}")
print(f"R²     : {r2_b:.3f}")
print(f"±2000  : {w2k_b*100:.1f}%")
print(f"Acc rounded (1000 UI): {acc_round_b*100:.1f}%")


# =============================================================================
# 10) MoE: 3 EXPERT XGB + MIX weighted with probs SAINT
# =============================================================================
experts = {}
for k in [0, 1, 2]:
    sub = train_all[train_all["Target_class"] == k].copy()
    print(f"\n--- Expert {k} | righe train: {len(sub)} | pazienti: {sub[ID_COL].nunique()} ---")

    
    if sub[ID_COL].nunique() < 10 or len(sub) < 200:
        print("   [WARN] small subset -> use train_all as a fallback for this expert.")
        sub = train_all.copy()

    booster_k, scaler_y_k = fit_xgb_with_internal_val(sub, XGB_PARAMS)
    experts[k] = (booster_k, scaler_y_k)


pred0 = predict_rel(experts[0][0], experts[0][1], test_all)
pred1 = predict_rel(experts[1][0], experts[1][1], test_all)
pred2 = predict_rel(experts[2][0], experts[2][1], test_all)


p0 = probs_te[:, 0].astype(np.float32)
p1 = probs_te[:, 1].astype(np.float32)
p2 = probs_te[:, 2].astype(np.float32)

pred_rel_moe = np.clip(p0 * pred0 + p1 * pred1 + p2 * pred2, -0.5, 0.5).astype(np.float32)

mae_m, rmse_m, r2_m, w2k_m, acc_round_m = eval_ui(test_all, pred_rel_moe)

print("\n==================== MoE (SAINT gate + 3 XGB experts) ====================")
print(f"MAE UI : {mae_m:.0f}")
print(f"RMSE UI: {rmse_m:.0f}")
print(f"R²     : {r2_m:.3f}")
print(f"±2000  : {w2k_m*100:.1f}%")
print(f"Acc rounded (1000 UI): {acc_round_m*100:.1f}%")


print("\n==================== DELTA (MoE vs Baseline) ====================")
print(f"ΔMAE UI : {mae_b - mae_m:.0f}")
print(f"ΔRMSE UI: {rmse_b - rmse_m:.0f}")
print(f"ΔR²     : {r2_m - r2_b:.3f}")
print(f"Δ±2000  : {(w2k_m - w2k_b)*100:.2f}%")
print(f"ΔAcc rounded : {(acc_round_m - acc_round_b)*100:.2f}%")

