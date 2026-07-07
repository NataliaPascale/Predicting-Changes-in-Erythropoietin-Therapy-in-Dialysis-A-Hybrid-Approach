import pandas as pd
import numpy as np
import lightgbm as lgb

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

SELECT_METRIC = "disc_mae"

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

if 'ERI' in df.columns:
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
      .transform(lambda x: x.rolling(5, min_periods=3).apply(hb_trend, raw=False))
)

df['EPO_Response'] = (df['Delta_Emoglobina'] / (df['Dose_lag_1'] + 1.0)).clip(-2, 2)
df['Hb_per_EPO']   = (df['Emoglobina'] / (df['Dose_lag_1'] + 1.0)).clip(0, 30)

if 'Ferritina' in df.columns and 'T-SAT' in df.columns:
    df['Ferritin_TSAT_ratio'] = (df['Ferritina'] / (df['T-SAT'] + 1.0)).clip(0, 200)

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
    'Hb_per_EPO','EPO_Response',
    'Emoglobina','Delta_Emoglobina',
    'Hb_std_3','Hb_Trend_5',
    'Dose_lag_1','Dose_lag_2',
    'ERI_lag_1','ERI_lag_2',
    'Ferritina','T-SAT','Transferrina',
    'Ferritin_TSAT_ratio',
    'Creatinina','Delta_Creatinina',
    'Peso Pre','Delta_Peso Pre',
    'Vitamina B12',
    'Dose_var_3'
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

# droppo SOLO il necessario
df_model = df.dropna(subset=['Dose_lag_1', 'Target_rel']).copy()

# =============================================================================
# 5. FIT 
# =============================================================================
def fit_lgb_with_internal_val(train_df_full, params):
    gss_val = GroupShuffleSplit(n_splits=1, test_size=VAL_SPLIT_IN_TRAIN, random_state=RANDOM_STATE)
    tr_idx, val_idx = next(gss_val.split(train_df_full, groups=train_df_full[ID_COL]))

    train_df = train_df_full.iloc[tr_idx].copy()
    val_df   = train_df_full.iloc[val_idx].copy()

    scaler_y = StandardScaler()

    X_train = train_df[FEATURE_COLS].values
    X_val   = val_df[FEATURE_COLS].values

    y_train = scaler_y.fit_transform(train_df[['Target_rel']].values).reshape(-1)
    y_val   = scaler_y.transform(val_df[['Target_rel']].values).reshape(-1)

    lgb_train = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
    lgb_val   = lgb.Dataset(X_val,   label=y_val, reference=lgb_train, free_raw_data=True)

    booster = lgb.train(
        params=params,
        train_set=lgb_train,
        num_boost_round=params.get("num_boost_round", 4000),
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train','val'],
        callbacks=[lgb.early_stopping(stopping_rounds=params.get("early_stopping_rounds", 80), verbose=False)]
    )

    best_it = int(booster.best_iteration or params.get("num_boost_round", 4000))
    return booster, scaler_y, best_it

# =============================================================================
# 6. EVAL 
# =============================================================================
def eval_on_df_lgb(booster, scaler_y, df_ref, TH_REL=0.15, TH_UI=1000):
    X = df_ref[FEATURE_COLS].values

    true_rel = df_ref[['Target_rel']].values  # non scalato
    dose_prev = df_ref['Dose_lag_1'].values.reshape(-1, 1)
    true_dose = np.maximum(0, dose_prev * (1 + true_rel))

    pred_scaled = booster.predict(X, num_iteration=booster.best_iteration).reshape(-1, 1)
    pred_rel = scaler_y.inverse_transform(pred_scaled)
    pred_rel = np.clip(pred_rel, -0.5, 0.5)


    pred_dose = np.maximum(0, dose_prev * (1 + pred_rel))

    mae = mean_absolute_error(true_dose, pred_dose)
    rmse = np.sqrt(mean_squared_error(true_dose, pred_dose))
    r2 = r2_score(true_dose, pred_dose)
    within_2k = float(np.mean(np.abs(pred_dose - true_dose) <= 2000))
    pred_round = np.round(pred_dose / 1000) * 1000
    acc_round = float(np.mean(np.abs(pred_round - true_dose) <= 2000))

  
    delta_ui = dose_prev * pred_rel
    pred_rel_d = pred_rel.copy()
    mask_up   = (pred_rel >= TH_REL)  & (delta_ui >= TH_UI)
    mask_down = (pred_rel <= -TH_REL) & (delta_ui <= -TH_UI)
    pred_rel_d[~(mask_up | mask_down)] = 0.0

    pred_dose_d = np.maximum(0, dose_prev * (1 + pred_rel_d))

    mae_d = mean_absolute_error(true_dose, pred_dose_d)
    rmse_d = np.sqrt(mean_squared_error(true_dose, pred_dose_d))
    r2_d = r2_score(true_dose, pred_dose_d)
    within_2k_d = float(np.mean(np.abs(pred_dose_d - true_dose) <= 2000))
    pred_round_d = np.round(pred_dose_d / 1000) * 1000
    acc_round_d = float(np.mean(np.abs(pred_round_d - true_dose) <= 2000))

    return {
        "cont": {"mae": float(mae), "rmse": float(rmse), "r2": float(r2),
                 "within2k": within_2k, "acc_round": acc_round},
        "disc": {"mae": float(mae_d), "rmse": float(rmse_d), "r2": float(r2_d),
                 "within2k": within_2k_d, "acc_round": acc_round_d},
    }

def pick_score(out, metric):
    if metric == "disc_mae":
        return out["disc"]["mae"]        
    if metric == "cont_mae":
        return out["cont"]["mae"]          
    raise ValueError("SELECT_METRIC not supported")

# =============================================================================
# 7. SPLIT TRAIN/TEST
# =============================================================================
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
tr_idx, te_idx = next(gss.split(df_model, groups=df_model[ID_COL]))

train_all = df_model.iloc[tr_idx].copy()
test_all  = df_model.iloc[te_idx].copy()

print("\n==================== SPLIT FINALE ====================")
print("Patients TRAIN:", train_all[ID_COL].nunique(), "| Rows TRAIN:", len(train_all))
print("Patients TEST :", test_all[ID_COL].nunique(),  "| Rows TEST :", len(test_all))

# =============================================================================
# 8. CV 
# =============================================================================
param_list = [
    
    {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': -1,
        'min_data_in_leaf': 40,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'lambda_l2': 2.0,
        'lambda_l1': 0.5,
        'verbosity': -1,
        'num_boost_round': 4000,
        'early_stopping_rounds': 80,
        'seed': RANDOM_STATE
    },
   
    {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': -1,
        'min_data_in_leaf': 60,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'lambda_l2': 4.0,
        'lambda_l1': 1.0,
        'verbosity': -1,
        'num_boost_round': 5000,
        'early_stopping_rounds': 120,
        'seed': RANDOM_STATE
    },
  
    {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': -1,
        'min_data_in_leaf': 30,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l2': 2.0,
        'lambda_l1': 0.5,
        'verbosity': -1,
        'num_boost_round': 4000,
        'early_stopping_rounds': 80,
        'seed': RANDOM_STATE
    }


]

gkf = GroupKFold(n_splits=N_SPLITS)

best_params = None
best_cv = np.inf

print("\n==================== CV ====================")
for i, params in enumerate(param_list, start=1):
    fold_scores = []

    for fold, (tri, vei) in enumerate(gkf.split(train_all, groups=train_all[ID_COL]), start=1):
        tr_df = train_all.iloc[tri].copy()
        va_df = train_all.iloc[vei].copy()

        booster, scaler_y, best_it = fit_lgb_with_internal_val(tr_df, params)
        out = eval_on_df_lgb(booster, scaler_y, va_df, TH_REL=TH_REL, TH_UI=TH_UI)

        fold_scores.append(pick_score(out, SELECT_METRIC))

    mean_score = float(np.mean(fold_scores))
    print(f"Config {i}/{len(param_list)} | {SELECT_METRIC}={mean_score:.0f} UI | lr={params['learning_rate']}, leaves={params['num_leaves']}, min_leaf={params['min_data_in_leaf']}")

    if mean_score < best_cv:
        best_cv = mean_score
        best_params = params

print("\n==================== BEST ====================")
print(f"Best {SELECT_METRIC}: {best_cv:.0f} UI")
print("Best params:", best_params)

# =============================================================================
# 9.Final Fit with best_params
# =============================================================================
final_booster, final_scaler_y, final_best_it = fit_lgb_with_internal_val(train_all, best_params)

out_train = eval_on_df_lgb(final_booster, final_scaler_y, train_all, TH_REL=TH_REL, TH_UI=TH_UI)
out_test  = eval_on_df_lgb(final_booster, final_scaler_y, test_all,  TH_REL=TH_REL, TH_UI=TH_UI)

print("\n==================== FINAL TRAIN (CONTINUOUS VALUE) ====================")
print(f"MAE: {out_train['cont']['mae']:.0f} UI")
print(f"RMSE: {out_train['cont']['rmse']:.0f} UI")
print(f"R²: {out_train['cont']['r2']:.3f}")
print(f"±2000 UI accuracy: {out_train['cont']['within2k']*100:.1f}%")
print(f"Acc rounded: {out_train['cont']['acc_round']*100:.1f}%")

print("\n==================== FINAL TRAIN (Discrete Threshold) ====================")
print(f"MAE: {out_train['disc']['mae']:.0f} UI")
print(f"RMSE: {out_train['disc']['rmse']:.0f} UI")
print(f"R²: {out_train['disc']['r2']:.3f}")
print(f"±2000 UI accuracy: {out_train['disc']['within2k']*100:.1f}%")
print(f"Acc rounded: {out_train['disc']['acc_round']*100:.1f}%")

print("\n==================== FINAL TEST (CONTINUOUS VALUE) ====================")
print(f"MAE: {out_test['cont']['mae']:.0f} UI")
print(f"RMSE: {out_test['cont']['rmse']:.0f} UI")
print(f"R²: {out_test['cont']['r2']:.3f}")
print(f"±2000 UI accuracy: {out_test['cont']['within2k']*100:.1f}%")
print(f"Acc rounded: {out_test['cont']['acc_round']*100:.1f}%")

print("\n==================== FINAL TEST (Discrete Threshold) ====================")
print(f"Soglie: TH_REL={TH_REL:.2f} | TH_UI={TH_UI:.0f}")
print(f"MAE: {out_test['disc']['mae']:.0f} UI")
print(f"RMSE: {out_test['disc']['rmse']:.0f} UI")
print(f"R²: {out_test['disc']['r2']:.3f}")
print(f"±2000 UI accuracy: {out_test['disc']['within2k']*100:.1f}%")
print(f"Acc rounded: {out_test['disc']['acc_round']*100:.1f}%")

