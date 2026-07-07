import pandas as pd
import numpy as np
import tensorflow as tf
import random, os

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Masking
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# =============================================================================
# 1. CONFIG
# =============================================================================
FILE_PATH = #Add the dataset path here

ID_COL = 'id_AnaPaz'
DATE_COL = 'DataReferto'
TARGET_COL = 'DosaggioSettimanaleTotale'
RANDOM_STATE = 42
MAX_SEQ_LEN = 5
PADDING_VALUE = -99.0

TEST_SIZE = 0.20
N_SPLITS_CV = 5
VAL_SPLIT_IN_TRAIN = 0.15

MAX_LAG_DOSE = 2
MAX_LAG_ERI  = 2
DELTA_COLS = ['Emoglobina', 'Ferritina', 'Creatinina', 'Peso Pre', 'T-SAT']

TH_REL = 0.15
TH_UI  = 1000

SELECT_METRIC = "disc_mae"  

# =============================================================================
# 2. LOAD + CLEAN
# =============================================================================
df = pd.read_excel(FILE_PATH)

df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors='coerce')
df = df.dropna(subset=[ID_COL, DATE_COL]).copy()
df = df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)


df = df.dropna(subset=['Emoglobina', 'Peso Pre', TARGET_COL]).copy()


other_features = ['Creatinina', 'Ferritina', 'T-SAT', 'Vitamina B12', 'Transferrina', 'ERI']
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
      .transform(lambda x: x.rolling(5, min_periods=3).apply(hb_trend, raw=False))
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

# =============================================================================
# 4. FINAL FEATURE
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
    'Hb_Trend_5',
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

# droppo solo il necessario
df_model = df.dropna(subset=['Dose_lag_1', 'Target_rel']).copy()

# =============================================================================
# 5. SEQUENCES
# =============================================================================
def create_variable_sequences(df_in, features, target, max_seq_len, padding_value):
    X_list, y_list = [], []
    for _, g in df_in.groupby(ID_COL):
        g = g.sort_values(DATE_COL)
        data = g[features].values
        target_values = g[target].values
        n = len(data)

        for i in range(n):
            start = max(0, i - max_seq_len + 1)
            seq = data[start:i+1]

            if len(seq) < max_seq_len:
                pad_len = max_seq_len - len(seq)
                pad = np.full((pad_len, seq.shape[1]), padding_value) 
                seq = np.vstack([pad, seq])

            X_list.append(seq)
            y_list.append(target_values[i])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

# =============================================================================
# 6. PREPROCESS (impute + isna + scaling) FIT on train_df
# =============================================================================
def preprocess_fit_transform(train_df, val_df, test_df, base_feature_cols):
  
    train_df = train_df.copy()
    val_df   = val_df.copy() if val_df is not None else None
    test_df  = test_df.copy() if test_df is not None else None

   
    ISNA_COLS = [
        'Ferritina', 'T-SAT', 'Transferrina', 'Ferritin_TSAT_ratio',
        'Creatinina', 'Vitamina B12',
        'Hb_std_3', 'Hb_Volatility', 'Hb_Trend_5', 'Dose_var_3'
    ]
    ISNA_COLS = [c for c in ISNA_COLS if c in base_feature_cols]

    for c in ISNA_COLS:
        train_df[c + "_isna"] = train_df[c].isna().astype(int)
        if val_df is not None:
            val_df[c + "_isna"] = val_df[c].isna().astype(int)
        if test_df is not None:
            test_df[c + "_isna"] = test_df[c].isna().astype(int)

    FEATURE_COLS_FINAL = base_feature_cols + [c + "_isna" for c in ISNA_COLS]

    
    fill_values = {c: train_df[c].median() for c in base_feature_cols}
    for c, fv in fill_values.items():
        train_df[c] = train_df[c].fillna(fv)
        if val_df is not None:
            val_df[c] = val_df[c].fillna(fv)
        if test_df is not None:
            test_df[c] = test_df[c].fillna(fv)

    # Scaling FIT train - Z-score
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    train_df[FEATURE_COLS_FINAL] = scaler_X.fit_transform(train_df[FEATURE_COLS_FINAL])
    if val_df is not None:
        val_df[FEATURE_COLS_FINAL]   = scaler_X.transform(val_df[FEATURE_COLS_FINAL])
    if test_df is not None:
        test_df[FEATURE_COLS_FINAL]  = scaler_X.transform(test_df[FEATURE_COLS_FINAL])

#scaler target
    train_df['Target_rel_scaled'] = scaler_y.fit_transform(train_df[['Target_rel']])
    if val_df is not None:
        val_df['Target_rel_scaled']   = scaler_y.transform(val_df[['Target_rel']])
    if test_df is not None:
        test_df['Target_rel_scaled']  = scaler_y.transform(test_df[['Target_rel']])

    return train_df, val_df, test_df, FEATURE_COLS_FINAL, scaler_X, scaler_y

# =============================================================================
# 7.EVAL
# =============================================================================
def eval_metrics_from_sequences(model, X, y_scaled, scaler_y, scaler_X, feature_cols_final,
                               TH_REL=0.15, TH_UI=1000):

    pred_scaled = model.predict(X, verbose=0)

    pred_rel = scaler_y.inverse_transform(pred_scaled)
    true_rel = scaler_y.inverse_transform(y_scaled.reshape(-1, 1))

    pred_rel = np.clip(pred_rel, -0.5, 0.5)

    last_step = scaler_X.inverse_transform(X[:, -1, :])
    dose_prev = last_step[:, feature_cols_final.index('Dose_lag_1')].reshape(-1, 1)

    true_dose = np.maximum(0, dose_prev * (1 + true_rel))

    pred_dose = np.maximum(0, dose_prev * (1 + pred_rel))

    cont_mae = mean_absolute_error(true_dose, pred_dose)
    cont_rmse = np.sqrt(mean_squared_error(true_dose, pred_dose))
    cont_r2 = r2_score(true_dose, pred_dose)
    cont_within2k = float(np.mean(np.abs(pred_dose - true_dose) <= 2000))
    pred_round = np.round(pred_dose / 1000) * 1000
    cont_acc_round = float(np.mean(np.abs(pred_round - true_dose) <= 2000))

    delta_ui = dose_prev * pred_rel

    pred_rel_discrete = pred_rel.copy()
    mask_up   = (pred_rel >= TH_REL)  & (delta_ui >= TH_UI)
    mask_down = (pred_rel <= -TH_REL) & (delta_ui <= -TH_UI)
    pred_rel_discrete[~(mask_up | mask_down)] = 0.0

    pred_dose_discrete = np.maximum(0, dose_prev * (1 + pred_rel_discrete))

    disc_mae = mean_absolute_error(true_dose, pred_dose_discrete)
    disc_rmse = np.sqrt(mean_squared_error(true_dose, pred_dose_discrete))
    disc_r2 = r2_score(true_dose, pred_dose_discrete)
    disc_within2k = float(np.mean(np.abs(pred_dose_discrete - true_dose) <= 2000))
    pred_round_d = np.round(pred_dose_discrete / 1000) * 1000
    disc_acc_round = float(np.mean(np.abs(pred_round_d - true_dose) <= 2000))

    return {
        "cont": {"mae": float(cont_mae), "rmse": float(cont_rmse), "r2": float(cont_r2),
                 "within2k": cont_within2k, "acc_round": cont_acc_round},
        "disc": {"mae": float(disc_mae), "rmse": float(disc_rmse), "r2": float(disc_r2),
                 "within2k": disc_within2k, "acc_round": disc_acc_round},
    }

# =============================================================================
# 8. PRINT
# =============================================================================
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
# 9. MODEL BUILDER
# =============================================================================

def build_lstm_model(n_features, cfg):
    model = Sequential([
        Input(shape=(MAX_SEQ_LEN, n_features)),
        Masking(mask_value=PADDING_VALUE),
        LSTM(cfg["lstm_units"], activation='tanh'),
        Dropout(cfg["dropout"]),
        Dense(cfg["dense_units"], activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["lr"]),
        loss=tf.keras.losses.Huber(delta=cfg["huber_delta"]),
        metrics=['mae']
    )
    return model

# =============================================================================
# 10. SPLIT TRAIN/TEST
# =============================================================================
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
tr_idx, te_idx = next(gss.split(df_model, groups=df_model[ID_COL]))

train_all = df_model.iloc[tr_idx].copy()
test_all  = df_model.iloc[te_idx].copy()

print("\n==================== FINAL SPLIT ====================")
print("Patients TRAIN:", train_all[ID_COL].nunique(), "| Rows TRAIN:", len(train_all))
print("Patients TEST :", test_all[ID_COL].nunique(),  "| Rows TEST :", len(test_all))

# =============================================================================
# 11. CV
# =============================================================================
lstm_param_list = [

    {"lstm_units": 64, "dense_units": 32, "dropout": 0.30, "lr": 5e-4, "huber_delta": 0.20,
     "batch_size": 32, "epochs": 120},

    # un po' più capiente
    {"lstm_units": 96, "dense_units": 48, "dropout": 0.35, "lr": 5e-4, "huber_delta": 0.20,
     "batch_size": 32, "epochs": 150},

    # più regolarizzato / più lento
    {"lstm_units": 64, "dense_units": 32, "dropout": 0.40, "lr": 3e-4, "huber_delta": 0.20,
     "batch_size": 32, "epochs": 160},
]

gkf = GroupKFold(n_splits=N_SPLITS_CV)

best_cfg = None
best_cv_score = np.inf


for i, cfg in enumerate(lstm_param_list, start=1):
    fold_scores = []
    
    for fold, (tri, vei) in enumerate(gkf.split(train_all, groups=train_all[ID_COL]), start=1):
        fold_train_full = train_all.iloc[tri].copy()
        fold_val_outer  = train_all.iloc[vei].copy()

        
        gss_val = GroupShuffleSplit(n_splits=1, test_size=VAL_SPLIT_IN_TRAIN, random_state=RANDOM_STATE)
        tr2_idx, val2_idx = next(gss_val.split(fold_train_full, groups=fold_train_full[ID_COL]))

        train_df = fold_train_full.iloc[tr2_idx].copy()
        val_int  = fold_train_full.iloc[val2_idx].copy()

        
        train_df_p, val_int_p, val_outer_p, feat_final, scaler_X, scaler_y = preprocess_fit_transform(
            train_df, val_int, fold_val_outer, FEATURE_COLS
        )

       
        X_train, y_train = create_variable_sequences(train_df_p, feat_final, 'Target_rel_scaled', MAX_SEQ_LEN, PADDING_VALUE)
        X_valint, y_valint = create_variable_sequences(val_int_p, feat_final, 'Target_rel_scaled', MAX_SEQ_LEN, PADDING_VALUE)
        X_valo, y_valo = create_variable_sequences(val_outer_p, feat_final, 'Target_rel_scaled', MAX_SEQ_LEN, PADDING_VALUE)

        
        model = build_lstm_model(n_features=len(feat_final), cfg=cfg)

        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True), 
            ReduceLROnPlateau(patience=5, factor=0.5)
        ]

        model.fit(
            X_train, y_train,
            validation_data=(X_valint, y_valint),
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            callbacks=callbacks,
            verbose=0
        )

        out = eval_metrics_from_sequences(
            model, X_valo, y_valo, scaler_y, scaler_X, feat_final,
            TH_REL=TH_REL, TH_UI=TH_UI
        )

        if SELECT_METRIC == "disc_mae":
            fold_scores.append(out["disc"]["mae"])
        else:
            fold_scores.append(out["cont"]["mae"])

    
        tf.keras.backend.clear_session()

    mean_score = float(np.mean(fold_scores))
    print(
        f"Config {i}/{len(lstm_param_list)} | {SELECT_METRIC}={mean_score:.0f} UI | "
        f"units={cfg['lstm_units']}, drop={cfg['dropout']}, lr={cfg['lr']}"
    )

    if mean_score < best_cv_score:
        best_cv_score = mean_score
        best_cfg = cfg

print("\n==================== BEST ====================")
print(f"Best {SELECT_METRIC}: {best_cv_score:.0f} UI")
print("Best cfg:", best_cfg)

# =============================================================================
# 12.Final Fit
# =============================================================================
gss_val_final = GroupShuffleSplit(n_splits=1, test_size=VAL_SPLIT_IN_TRAIN, random_state=RANDOM_STATE)
trf_idx, vaf_idx = next(gss_val_final.split(train_all, groups=train_all[ID_COL]))

train_df_final = train_all.iloc[trf_idx].copy()
val_int_final  = train_all.iloc[vaf_idx].copy()

train_df_p, val_int_p, test_p, feat_final, scaler_X, scaler_y = preprocess_fit_transform(
    train_df_final, val_int_final, test_all, FEATURE_COLS
)

X_train, y_train = create_variable_sequences(train_df_p, feat_final, 'Target_rel_scaled', MAX_SEQ_LEN, PADDING_VALUE)
X_val, y_val     = create_variable_sequences(val_int_p, feat_final, 'Target_rel_scaled', MAX_SEQ_LEN, PADDING_VALUE)
X_test, y_test   = create_variable_sequences(test_p, feat_final, 'Target_rel_scaled', MAX_SEQ_LEN, PADDING_VALUE)

print("\n==================== TRAIN FINALE ====================")
print("Sequences TRAIN:", X_train.shape)
print("Sequences VAL  :", X_val.shape)
print("Sequencs TEST :", X_test.shape)
print("Features :", len(feat_final))

final_model = build_lstm_model(n_features=len(feat_final), cfg=best_cfg)

callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True),
    ReduceLROnPlateau(patience=5, factor=0.5)
]

final_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=best_cfg["epochs"],
    batch_size=best_cfg["batch_size"],
    callbacks=callbacks,
    verbose=0
)

out_train = eval_metrics_from_sequences(final_model, X_train, y_train, scaler_y, scaler_X, feat_final, TH_REL=TH_REL, TH_UI=TH_UI)
out_test  = eval_metrics_from_sequences(final_model, X_test,  y_test,  scaler_y, scaler_X, feat_final, TH_REL=TH_REL, TH_UI=TH_UI)

print_eval("FINAL TRAIN", out_train)
print_eval("FINAL TEST",  out_test)

