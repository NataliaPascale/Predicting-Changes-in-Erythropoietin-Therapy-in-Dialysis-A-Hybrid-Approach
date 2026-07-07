import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import xgboost as xgb
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Masking
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau



# =============================================================================
# 0. CONFIG
# =============================================================================
FILE_PATH = #Add the dataset path here
ID_COL = 'id_AnaPaz'
DATE_COL = 'DataReferto'
TARGET_COL = 'DosaggioSettimanaleTotale'

TEST_SIZE = 0.20
RANDOM_STATE_SPLIT = 42


TH_REL = 0.15
TH_UI  = 1000

# feature engineering
MAX_LAG_DOSE = 2
MAX_LAG_ERI  = 2
DELTA_COLS = ['Emoglobina', 'Ferritina', 'Creatinina', 'Peso Pre', 'T-SAT']

# LSTM seq
MAX_SEQ_LEN = 5
PADDING_VALUE = -99.0

# training experts
VAL_SPLIT_IN_TRAIN = 0.15  
XGB_NUM_ROUNDS = 1200
XGB_EARLY_STOP = 50

LSTM_CFG = dict(
    lstm_units=96,
    dense_units=48,
    dropout=0.35,
    lr=5e-4,
    huber_delta=0.20,
    batch_size=32,
    epochs=170,
)

# ---- Path SAINT ----
SAINT_PATH = #Add the SAINT model path here
SAINT_W = f"{SAINT_PATH}/model_weights.pt"
SAINT_SCALER = f"{SAINT_PATH}/scaler.pkl"
SAINT_META = f"{SAINT_PATH}/metadata.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

# =============================================================================
# 1. LOAD + CLEAN + FEATURE ENGINEERING 
# =============================================================================
df = pd.read_excel(FILE_PATH)

df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors='coerce')
df = df.dropna(subset=[ID_COL, DATE_COL]).copy()
df = df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)


df = df.dropna(subset=['Emoglobina', 'Peso Pre', TARGET_COL]).copy()

other_features = ['Creatinina', 'Ferritina', 'T-SAT', 'Vitamina B12', 'Transferrina', 'ERI']
existing_cols = [c for c in other_features if c in df.columns]
df[existing_cols] = df.groupby(ID_COL)[existing_cols].ffill()

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

df['EPO_Response'] = (df.get('Delta_Emoglobina', 0) / (df['Dose_lag_1'] + 1.0)).clip(-2, 2)
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
# 2. TARGET CLASS (0=DOWN, 1=KEEP, 2=UP)
# =============================================================================
delta_ui_true = df_model['Dose_lag_1'] * df_model['Target_rel']

is_up   = (df_model['Target_rel'] >= TH_REL)  & (delta_ui_true >= TH_UI)
is_down = (df_model['Target_rel'] <= -TH_REL) & (delta_ui_true <= -TH_UI)

df_model['Target_class'] = 1
df_model.loc[is_down, 'Target_class'] = 0
df_model.loc[is_up,   'Target_class'] = 2

print("Rows:", len(df_model))
print("Class dist:", df_model['Target_class'].value_counts().sort_index().to_dict())

# =============================================================================
# 3. SPLIT TRAIN/TEST
# =============================================================================
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE_SPLIT)
tr_idx, te_idx = next(gss.split(df_model, groups=df_model[ID_COL]))
train_all = df_model.iloc[tr_idx].copy()
test_all  = df_model.iloc[te_idx].copy()

print("\n==================== FINAL SPLIT ====================")
print(f"Patients TRAIN: {train_all[ID_COL].nunique()} | Rows TRAIN: {len(train_all)}")
print(f"Patients TEST : {test_all[ID_COL].nunique()} | Rows TEST : {len(test_all)}")

# =============================================================================
# 4. SAINT + LOAD
# =============================================================================
class SAINT_NumOnly(nn.Module):
    def __init__(self, n_features, n_classes=3, d_model=128, n_heads=8, n_layers=4, ff_dim=256, dropout=0.10):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.proj = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_features)])
        self.pos = nn.Parameter(torch.zeros(1, 1 + n_features, d_model))

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
                dropout=dropout, batch_first=True
            ) for _ in range(n_layers)
        ])

        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

    def forward(self, x):
        B, F = x.shape
        cls = self.cls.expand(B, -1, -1)
        feats = torch.cat([self.proj[i](x[:, i:i+1]).unsqueeze(1) for i in range(F)], dim=1)
        z = torch.cat([cls, feats], dim=1) + self.pos[:, :F+1, :]
        for blk in self.blocks:
            z = blk(z)
        return self.head(z[:, 0, :])

meta = joblib.load(SAINT_META)
scaler_saint = joblib.load(SAINT_SCALER)

feature_final_saint = meta["feature_final"]
isna_cols_saint = meta["isna_cols"]
medians_saint = meta["medians"]
best = meta["best_params"]
n_feat_saint = int(meta["n_feat"])

print("\n[SAINT] feature_final len:", len(feature_final_saint))
print("[SAINT] best params:", best)

saint = SAINT_NumOnly(
    n_features=n_feat_saint,
    n_classes=3,
    d_model=int(best["d_model"]),
    n_heads=int(best["n_heads"]),
    n_layers=int(best["n_layers"]),
    ff_dim=256,
    dropout=float(best["dropout"]),
).to(DEVICE)

state = torch.load(SAINT_W, map_location=DEVICE)
saint.load_state_dict(state)
saint.eval()

# =============================================================================
# 5. PREP for SAINT: isna + imputation + scaling 
# =============================================================================
def prep_for_saint(df_in):
    dfp = df_in.copy()

    T
    for c in isna_cols_saint:
        if c in dfp.columns:
            dfp[c + "_isna"] = dfp[c].isna().astype(int)
        else:
            
            dfp[c] = np.nan
            dfp[c + "_isna"] = 1

    
    for c in medians_saint.index:
        if c in dfp.columns:
            dfp[c] = dfp[c].fillna(medians_saint[c])
        else:
            dfp[c] = medians_saint[c]

    
    for c in feature_final_saint:
        if c not in dfp.columns:
            dfp[c] = 0

    X = dfp[feature_final_saint].values
    Xs = scaler_saint.transform(X).astype(np.float32)
    return Xs

def saint_proba(df_in, batch=2048):
    Xs = prep_for_saint(df_in)
    probs = []
    with torch.no_grad():
        for i in range(0, len(Xs), batch):
            xb = torch.tensor(Xs[i:i+batch]).to(DEVICE)
            logits = saint(xb)
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)
    return np.vstack(probs)

# =============================================================================
# 6. EXPERTS: XGB 
# =============================================================================
XGB_PARAMS = dict(
    objective='reg:squarederror',
    eval_metric='rmse',
    eta=0.03,
    max_depth=3,
    min_child_weight=15,
    subsample=0.7,
    colsample_bytree=0.7,
    alpha=0.5,
    reg_lambda=2.0
)

def fit_xgb_expert(train_df_full):

    gss_val = GroupShuffleSplit(n_splits=1, test_size=VAL_SPLIT_IN_TRAIN, random_state=RANDOM_STATE_SPLIT)
    tr_i, va_i = next(gss_val.split(train_df_full, groups=train_df_full[ID_COL]))
    tr = train_df_full.iloc[tr_i].copy()
    va = train_df_full.iloc[va_i].copy()

    scaler_y = StandardScaler()
    y_tr = scaler_y.fit_transform(tr[['Target_rel']]).reshape(-1)
    y_va = scaler_y.transform(va[['Target_rel']]).reshape(-1)

    dtr = xgb.DMatrix(tr[FEATURE_COLS].values, label=y_tr, missing=np.nan)
    dva = xgb.DMatrix(va[FEATURE_COLS].values, label=y_va, missing=np.nan)

    booster = xgb.train(
        params=XGB_PARAMS,
        dtrain=dtr,
        num_boost_round=XGB_NUM_ROUNDS,
        evals=[(dtr,'train'), (dva,'val')],
        early_stopping_rounds=XGB_EARLY_STOP,
        verbose_eval=False
    )
    return booster, scaler_y

def xgb_predict_rel(booster, scaler_y, df_in):
    d = xgb.DMatrix(df_in[FEATURE_COLS].values, missing=np.nan)
    pred_s = booster.predict(d).reshape(-1, 1)
    pred = scaler_y.inverse_transform(pred_s)
    return np.clip(pred, -0.5, 0.5)  # shape (n,1)

# =============================================================================
# 7. EXPERTS: LSTM 
# =============================================================================
def create_sequences(df_in, features, target_scaled_col):
    X_list, y_list = [], []
    for _, g in df_in.groupby(ID_COL):
        g = g.sort_values(DATE_COL)
        data = g[features].values
        target_values = g[target_scaled_col].values
        n = len(data)

        for i in range(n):
            start = max(0, i - MAX_SEQ_LEN + 1)
            seq = data[start:i+1]
            if len(seq) < MAX_SEQ_LEN:
                pad = np.full((MAX_SEQ_LEN - len(seq), seq.shape[1]), PADDING_VALUE)
                seq = np.vstack([pad, seq])
            X_list.append(seq)
            y_list.append(target_values[i])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

def build_lstm(n_features):
    m = Sequential([
        Input(shape=(MAX_SEQ_LEN, n_features)),
        Masking(mask_value=PADDING_VALUE),
        LSTM(LSTM_CFG["lstm_units"], activation='tanh'),
        Dropout(LSTM_CFG["dropout"]),
        Dense(LSTM_CFG["dense_units"], activation='relu'),
        Dense(1)
    ])
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LSTM_CFG["lr"]),
        loss=tf.keras.losses.Huber(delta=LSTM_CFG["huber_delta"]),
        metrics=['mae']
    )
    return m

def fit_lstm_expert(train_df_full):
    gss_val = GroupShuffleSplit(n_splits=1, test_size=VAL_SPLIT_IN_TRAIN, random_state=RANDOM_STATE_SPLIT)
    tr_i, va_i = next(gss_val.split(train_df_full, groups=train_df_full[ID_COL]))
    tr = train_df_full.iloc[tr_i].copy()
    va = train_df_full.iloc[va_i].copy()

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    med = tr[FEATURE_COLS].median()
    for c in FEATURE_COLS:
        tr[c] = tr[c].fillna(med[c])
        va[c] = va[c].fillna(med[c])

    tr[FEATURE_COLS] = scaler_X.fit_transform(tr[FEATURE_COLS])
    va[FEATURE_COLS] = scaler_X.transform(va[FEATURE_COLS])

    tr['y_s'] = scaler_y.fit_transform(tr[['Target_rel']])
    va['y_s'] = scaler_y.transform(va[['Target_rel']])

    X_tr, y_tr = create_sequences(tr, FEATURE_COLS, 'y_s')
    X_va, y_va = create_sequences(va, FEATURE_COLS, 'y_s')

    model = build_lstm(n_features=len(FEATURE_COLS))

    cb = [
        EarlyStopping(patience=15, restore_best_weights=True),
        ReduceLROnPlateau(patience=5, factor=0.5)
    ]

    model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=LSTM_CFG["epochs"],
        batch_size=LSTM_CFG["batch_size"],
        verbose=0,
        callbacks=cb
    )
    return model, scaler_X, scaler_y, med

def lstm_predict_rel(model, scaler_X, scaler_y, med, df_in):
    d = df_in.copy()
    for c in FEATURE_COLS:
        d[c] = d[c].fillna(med[c])
    d[FEATURE_COLS] = scaler_X.transform(d[FEATURE_COLS])
    
    d['y_s'] = 0.0
    X, _ = create_sequences(d, FEATURE_COLS, 'y_s')
    pred_s = model.predict(X, verbose=0).reshape(-1, 1)
    pred = scaler_y.inverse_transform(pred_s)
    return np.clip(pred, -0.5, 0.5)

# =============================================================================
# 8. Train experts 
# =============================================================================
experts = {}
for cls in [0,1,2]:
    sub = train_all[train_all['Target_class'] == cls].copy()
    print(f"\n[EXPERT TRAIN] class={cls} rows={len(sub)} patients={sub[ID_COL].nunique()}")

    if len(sub) < 200:
        print("  -> Class too small; using train_all as a fallback.")
        sub = train_all.copy()

    # XGB
    xgb_boost, xgb_sy = fit_xgb_expert(sub)

    # LSTM
    lstm_model, lstm_sx, lstm_sy, lstm_med = fit_lstm_expert(sub)

    experts[cls] = dict(
        xgb=(xgb_boost, xgb_sy),
        lstm=(lstm_model, lstm_sx, lstm_sy, lstm_med),
    )

# =============================================================================
# 9. Mix XGB+LSTM
# =============================================================================
W_XGB = {0: 0.5, 1: 0.5, 2: 0.5}
W_LSTM = {0: 0.5, 1: 0.5, 2: 0.5}

# =============================================================================
# 10. Inference MoE on TEST: SAINT probs -> pred_rel for class -> mix
# =============================================================================
def moe_predict_rel(df_in, force_keep_rel0=True):
    # gating
    P = saint_proba(df_in)  # shape (n,3)

    # predictions for every class
    pred_by_cls = []
    for cls in [0,1,2]:
        xgb_boost, xgb_sy = experts[cls]["xgb"]
        lstm_model, lstm_sx, lstm_sy, lstm_med = experts[cls]["lstm"]

        pr_x = xgb_predict_rel(xgb_boost, xgb_sy, df_in)         # (n,1)
        pr_l = lstm_predict_rel(lstm_model, lstm_sx, lstm_sy, lstm_med, df_in)  # (n,1)
        pr = W_XGB[cls]*pr_x + W_LSTM[cls]*pr_l                  # (n,1)

        if force_keep_rel0 and cls == 1:
            pr[:] = 0.0

        pred_by_cls.append(pr)

    pred_by_cls = np.concatenate(pred_by_cls, axis=1)  # (n,3)

    # weighted sum
    pred_rel = (P * pred_by_cls).sum(axis=1, keepdims=True)
    pred_rel = np.clip(pred_rel, -0.5, 0.5)
    return pred_rel, P, pred_by_cls

# =============================================================================
# 11. Metrics
# =============================================================================
def eval_rel_to_dose(df_ref, pred_rel, TH_REL=0.15, TH_UI=1000):
    true_rel = df_ref[['Target_rel']].values
    dose_prev = df_ref['Dose_lag_1'].values.reshape(-1, 1)

    true_dose = np.maximum(0, dose_prev * (1 + true_rel))

  
    pred_dose = np.maximum(0, dose_prev * (1 + pred_rel))
    cont_mae = mean_absolute_error(true_dose, pred_dose)
    cont_rmse = np.sqrt(mean_squared_error(true_dose, pred_dose))
    cont_r2 = r2_score(true_dose, pred_dose)
    cont_within2k = float(np.mean(np.abs(pred_dose - true_dose) <= 2000))
    pred_round = np.round(pred_dose / 1000) * 1000
    cont_acc_round = float(np.mean(np.abs(pred_round - true_dose) <= 2000))

    
    delta_ui = dose_prev * pred_rel
    pred_rel_d = pred_rel.copy()
    mask_up   = (pred_rel >= TH_REL)  & (delta_ui >= TH_UI)
    mask_down = (pred_rel <= -TH_REL) & (delta_ui <= -TH_UI)
    pred_rel_d[~(mask_up | mask_down)] = 0.0

    pred_dose_d = np.maximum(0, dose_prev * (1 + pred_rel_d))
    disc_mae = mean_absolute_error(true_dose, pred_dose_d)
    disc_rmse = np.sqrt(mean_squared_error(true_dose, pred_dose_d))
    disc_r2 = r2_score(true_dose, pred_dose_d)
    disc_within2k = float(np.mean(np.abs(pred_dose_d - true_dose) <= 2000))
    pred_round_d = np.round(pred_dose_d / 1000) * 1000
    disc_acc_round = float(np.mean(np.abs(pred_round_d - true_dose) <= 2000))

    return dict(
        cont=dict(mae=float(cont_mae), rmse=float(cont_rmse), r2=float(cont_r2),
                  within2k=cont_within2k, acc_round=cont_acc_round),
        disc=dict(mae=float(disc_mae), rmse=float(disc_rmse), r2=float(disc_r2),
                  within2k=disc_within2k, acc_round=disc_acc_round),
    )

def print_eval(name, out):
    print(f"\n==================== {name} (CONTINUOUS VALUE) ====================")
    print(f"MAE: {out['cont']['mae']:.0f} UI")
    print(f"RMSE: {out['cont']['rmse']:.0f} UI")
    print(f"R²: {out['cont']['r2']:.3f}")
    print(f"±2000 UI accuracy: {out['cont']['within2k']*100:.1f}%")
    print(f"Acc rounded: {out['cont']['acc_round']*100:.1f}%")

    print(f"\n==================== {name} (DISCRETE THRESHOLD) ====================")
    print(f"MAE: {out['disc']['mae']:.0f} UI")
    print(f"RMSE: {out['disc']['rmse']:.0f} UI")
    print(f"R²: {out['disc']['r2']:.3f}")
    print(f"±2000 UI accuracy: {out['disc']['within2k']*100:.1f}%")
    print(f"Acc rounded: {out['disc']['acc_round']*100:.1f}%")

# =============================================================================
# 12. RUN su TEST
# =============================================================================
pred_rel, P, pred_by_cls = moe_predict_rel(test_all, force_keep_rel0=True)
out_test = eval_rel_to_dose(test_all, pred_rel, TH_REL=TH_REL, TH_UI=TH_UI)

print_eval("MOE TEST (SAINT gating + XGB/LSTM experts)", out_test)

