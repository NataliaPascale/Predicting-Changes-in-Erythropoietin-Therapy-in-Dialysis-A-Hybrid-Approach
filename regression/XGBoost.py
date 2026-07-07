import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# 1. CONFIG
# =============================================================================
FILE_PATH = #Add the dataset path here

ID_COL = 'id_AnaPaz'
DATE_COL = 'DataReferto'
TARGET_COL = 'DosaggioSettimanaleTotale'

N_SPLITS = 5
VAL_SPLIT_IN_TRAIN = 0.15
TEST_SIZE = 0.20
RANDOM_STATE = 42

TH_REL = 0.15
TH_UI  = 1000

# =============================================================================
# 2. LOAD + PREPROCESS
# =============================================================================
df = pd.read_excel(FILE_PATH)

df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors='coerce')
df = df.dropna(subset=[ID_COL, DATE_COL]).copy()
df = df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)

df = df.dropna(subset=['Emoglobina', 'Peso Pre', TARGET_COL]).copy()

fill_cols = ['Creatinina', 'Ferritina', 'T-SAT', 'Vitamina B12', 'Transferrina', 'ERI']
fill_cols = [c for c in fill_cols if c in df.columns]
df[fill_cols] = df.groupby(ID_COL)[fill_cols].ffill()

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================
df['Dose_lag_1'] = df.groupby(ID_COL)[TARGET_COL].shift(1)
df['Dose_lag_2'] = df.groupby(ID_COL)[TARGET_COL].shift(2)

df['ERI_lag_1'] = df.groupby(ID_COL)['ERI'].shift(1)
df['ERI_lag_2'] = df.groupby(ID_COL)['ERI'].shift(2)

for c in ['Emoglobina', 'Ferritina', 'Creatinina', 'Peso Pre', 'T-SAT']:
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

df['EPO_Response'] = (df['Delta_Emoglobina'] / (df['Dose_lag_1'] + 1)).clip(-2, 2)
df['Hb_per_EPO']   = (df['Emoglobina'] / (df['Dose_lag_1'] + 1)).clip(0, 30)

if 'Ferritina' in df.columns and 'T-SAT' in df.columns:
    df['Ferritin_TSAT_ratio'] = (df['Ferritina'] / (df['T-SAT'] + 1)).clip(0, 200)

df['Dose_var_3'] = (
    df.groupby(ID_COL)[TARGET_COL]
      .rolling(3).std()
      .reset_index(level=0, drop=True)
).clip(0, 10000)

df['Target_rel'] = (df[TARGET_COL] - df['Dose_lag_1']) / (df['Dose_lag_1'] + 1e-6)
df['Target_rel'] = df['Target_rel'].clip(-0.5, 0.5)

# =============================================================================
# 4. FINAL FEATURES
# =============================================================================
FEATURE_COLS = [
    'Hb_per_EPO', 'EPO_Response',
    'Emoglobina', 'Delta_Emoglobina',
    'Hb_std_3', 'Hb_Trend_5',
    'Dose_lag_1', 'Dose_lag_2',
    'ERI_lag_1', 'ERI_lag_2',
    'Ferritina', 'T-SAT', 'Transferrina',
    'Ferritin_TSAT_ratio',
    'Creatinina', 'Delta_Creatinina',
    'Peso Pre', 'Delta_Peso Pre',
    'Vitamina B12',
    'Dose_var_3'
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

df_model = df.dropna(subset=['Dose_lag_1', 'Target_rel']).copy()

# =============================================================================
# 5. Function TRAIN + EVAL
# =============================================================================
def fit_xgb_with_internal_val(train_df_full, params):
    gss_val = GroupShuffleSplit(n_splits=1, test_size=VAL_SPLIT_IN_TRAIN, random_state=RANDOM_STATE)
    tr_idx, val_idx = next(gss_val.split(train_df_full, groups=train_df_full[ID_COL]))

    train_df = train_df_full.iloc[tr_idx].copy()
    val_df   = train_df_full.iloc[val_idx].copy()

    scaler_y = StandardScaler()

    X_train = train_df[FEATURE_COLS].values
    X_val   = val_df[FEATURE_COLS].values
    y_train = train_df[['Target_rel']].values
    y_val   = val_df[['Target_rel']].values

    y_train_s = scaler_y.fit_transform(y_train).reshape(-1)
    y_val_s   = scaler_y.transform(y_val).reshape(-1)

    dtrain = xgb.DMatrix(X_train, label=y_train_s, missing=np.nan)
    dval   = xgb.DMatrix(X_val,   label=y_val_s,   missing=np.nan)

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=800,
        evals=[(dtrain,'train'), (dval,'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    return booster, scaler_y

def eval_on_df(booster, scaler_y, df_ref, TH_REL=0.15, TH_UI=1000):
    X = df_ref[FEATURE_COLS].values
    dmat = xgb.DMatrix(X, missing=np.nan)

   
    pred_scaled = booster.predict(dmat).reshape(-1, 1)
    pred_rel = scaler_y.inverse_transform(pred_scaled)
    pred_rel = np.clip(pred_rel, -0.5, 0.5)

    true_rel = df_ref[['Target_rel']].values 

    dose_prev = df_ref['Dose_lag_1'].values.reshape(-1, 1)
    true_dose = np.maximum(0, dose_prev * (1 + true_rel))

    
    pred_dose = np.maximum(0, dose_prev * (1 + pred_rel))

    mae = mean_absolute_error(true_dose, pred_dose)
    rmse = np.sqrt(mean_squared_error(true_dose, pred_dose))
    r2 = r2_score(true_dose, pred_dose)
    within_2k = np.mean(np.abs(pred_dose - true_dose) <= 2000)

    
    pred_round = np.round(pred_dose / 1000) * 1000
    acc_round = np.mean(np.abs(pred_round - true_dose) <= 2000)

    
    delta_ui = dose_prev * pred_rel 

    pred_rel_discrete = pred_rel.copy()
    mask_up   = (pred_rel >= TH_REL)  & (delta_ui >= TH_UI)
    mask_down = (pred_rel <= -TH_REL) & (delta_ui <= -TH_UI)

    pred_rel_discrete[~(mask_up | mask_down)] = 0.0
    pred_dose_discrete = np.maximum(0, dose_prev * (1 + pred_rel_discrete))

    mae_d = mean_absolute_error(true_dose, pred_dose_discrete)
    rmse_d = np.sqrt(mean_squared_error(true_dose, pred_dose_discrete))
    r2_d = r2_score(true_dose, pred_dose_discrete)
    within_2k_d = np.mean(np.abs(pred_dose_discrete - true_dose) <= 2000)

    pred_round_d = np.round(pred_dose_discrete / 1000) * 1000
    acc_round_d = np.mean(np.abs(pred_round_d - true_dose) <= 2000)

    return {
        "cont":  {"mae": mae,  "rmse": rmse,  "r2": r2,  "within2k": within_2k,  "acc_round": acc_round},
        "disc":  {"mae": mae_d,"rmse": rmse_d,"r2": r2_d,"within2k": within_2k_d,"acc_round": acc_round_d},
    }


# =============================================================================
# 6) SPLIT TRAIN/TEST
# =============================================================================
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
tr_idx, te_idx = next(gss.split(df_model, groups=df_model[ID_COL]))

train_all = df_model.iloc[tr_idx].copy()
test_all  = df_model.iloc[te_idx].copy()

print("\n==================== SPLIT FINALE ====================")
print("Patients TRAIN:", train_all[ID_COL].nunique(), "| Row TRAIN:", len(train_all))
print("Patients TEST :", test_all[ID_COL].nunique(),  "| Row TEST :", len(test_all))

# =============================================================================
# 7) CV 
# =============================================================================
param_list = [
    
    {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.03,
        'max_depth': 3,
        'min_child_weight': 15,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'alpha': 0.5,
        'lambda': 2.0
    },

    {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.05,
        'max_depth': 4,
        'min_child_weight': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'alpha': 0.5,
        'lambda': 2.0
    },
    {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.02,
        'max_depth': 3,
        'min_child_weight': 20,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'alpha': 1.0,
        'lambda': 2.0
    },
]

gkf = GroupKFold(n_splits=N_SPLITS)

SELECT_METRIC = "disc_mae"  

best_params = None
best_cv_score = np.inf


for i, params in enumerate(param_list, start=1):
    fold_scores = []

    for fold, (tri, vei) in enumerate(gkf.split(train_all, groups=train_all[ID_COL]), start=1):
        tr_df = train_all.iloc[tri].copy()
        va_df = train_all.iloc[vei].copy()

        booster, scaler_y = fit_xgb_with_internal_val(tr_df, params)
        out = eval_on_df(booster, scaler_y, va_df, TH_REL=TH_REL, TH_UI=TH_UI)

        if SELECT_METRIC == "disc_mae":
            fold_scores.append(out["disc"]["mae"])
        else:
            fold_scores.append(out["cont"]["mae"])

    mean_score = float(np.mean(fold_scores))
    print(f"Config {i}/{len(param_list)} | {SELECT_METRIC}={mean_score:.0f} UI | eta={params['eta']}, depth={params['max_depth']}, mcw={params['min_child_weight']}")

    if mean_score < best_cv_score:
        best_cv_score = mean_score
        best_params = params

print("\n==================== BEST (from CV on TRAIN) ====================")
print(f"Best {SELECT_METRIC}: {best_cv_score:.0f} UI")
print("Best params:", best_params)

# =============================================================================
# 8) Final Fit with best_params
# =============================================================================
final_booster, final_scaler_y = fit_xgb_with_internal_val(train_all, best_params)

out_test = eval_on_df(final_booster, final_scaler_y, test_all, TH_REL=TH_REL, TH_UI=TH_UI)

print("\n==================== FINAL TEST (CONTINUOUS VALUE) ====================")
print(f"MAE: {out_test['cont']['mae']:.0f} UI")
print(f"RMSE: {out_test['cont']['rmse']:.0f} UI")
print(f"R²: {out_test['cont']['r2']:.3f}")
print(f"±2000 UI accuracy: {out_test['cont']['within2k']*100:.1f}%")
print(f"Acc rounded: {out_test['cont']['acc_round']*100:.1f}%")

print("\n==================== FINAL TEST (With discrete threshold) ====================")
print(f"MAE: {out_test['disc']['mae']:.0f} UI")
print(f"RMSE: {out_test['disc']['rmse']:.0f} UI")
print(f"R²: {out_test['disc']['r2']:.3f}")
print(f"±2000 UI accuracy: {out_test['disc']['within2k']*100:.1f}%")
print(f"Acc rounded: {out_test['disc']['acc_round']*100:.1f}%")

