import pandas as pd
import numpy as np
import tensorflow as tf
import random, os

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Masking
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf



# =============================================================================
# 1. CONFIG
# =============================================================================
FILE_PATH = #Add the dataset path here

ID_COL = 'id_AnaPaz'
DATE_COL = 'DataReferto'
TARGET_COL = 'DosaggioSettimanaleTotale'

MAX_SEQ_LEN = 5
PADDING_VALUE = -99.0
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS_CV = 5
VAL_SPLIT_IN_TRAIN = 0.15

MAX_LAG_DOSE = 2
MAX_LAG_ERI  = 2
DELTA_COLS = ['Emoglobina', 'Ferritina', 'Creatinina', 'Peso Pre', 'T-SAT']


TH_REL = 0.15
TH_UI  = 1000


SELECT_METRIC = "macro_f1"

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

# Target rel
df['Target_rel'] = (df[TARGET_COL] - df['Dose_lag_1']) / (df['Dose_lag_1'] + 1e-6)
df['Target_rel'] = df['Target_rel'].clip(-0.5, 0.5)

# =============================================================================
# 4. Final Features
# =============================================================================
FEATURE_COLS = [
    'Hb_per_EPO','EPO_Response',
    'Emoglobina','Delta_Emoglobina',
    'Hb_std_3','Hb_Volatility','Hb_Trend_5',
    'Dose_lag_1','Dose_lag_2',
    'ERI_lag_1','ERI_lag_2',
    'Dose_var_3',
    'Ferritina','T-SAT','Transferrina','Ferritin_TSAT_ratio',
    'Creatinina','Delta_Creatinina',
    'Peso Pre','Delta_Peso Pre',
    'Vitamina B12',
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]


df_model = df.dropna(subset=['Dose_lag_1', 'Target_rel']).copy()

# =============================================================================
# 5. COSTRUCTION LABELS (3 CLASS-->0=DOWN, 1=KEEP, 2=UP)
# =============================================================================
def make_labels(df_in, th_rel=0.15, th_ui=1000):
    dose_prev = df_in['Dose_lag_1'].values
    rel = df_in['Target_rel'].values
    delta_ui = dose_prev * rel

    y = np.ones(len(df_in), dtype=np.int64)  # default KEEP=1
    up_mask   = (rel >=  th_rel) & (delta_ui >=  th_ui)
    down_mask = (rel <= -th_rel) & (delta_ui <= -th_ui)

    y[down_mask] = 0
    y[up_mask]   = 2
    return y

df_model['y_class'] = make_labels(df_model, TH_REL, TH_UI)

# =============================================================================
# 6. SEQUENCE (X) + label 
# =============================================================================
def create_sequences_with_labels(df_in, features, label_col, max_seq_len, padding_value):
    X_list, y_list = [], []

    for _, g in df_in.groupby(ID_COL):
        g = g.sort_values(DATE_COL)
        data = g[features].values
        labels = g[label_col].values
        n = len(g)

        for i in range(n):
            start = max(0, i - max_seq_len + 1)
            seq = data[start:i+1]

            if len(seq) < max_seq_len:
                pad_len = max_seq_len - len(seq)
                pad = np.full((pad_len, seq.shape[1]), padding_value)
                seq = np.vstack([pad, seq])

            X_list.append(seq)
            y_list.append(labels[i])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y

# =============================================================================
# 7. PREPROCESS (impute + isna + scaling)
# =============================================================================
def preprocess_fit_transform(train_df, val_df, test_df, base_feature_cols):
    train_df = train_df.copy()
    val_df   = val_df.copy() if val_df is not None else None
    test_df  = test_df.copy() if test_df is not None else None

    # Missing indicators (selettive)
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

    feature_cols_final = base_feature_cols + [c + "_isna" for c in ISNA_COLS]

    # Imputation
    fill_values = {c: train_df[c].median() for c in base_feature_cols}
    for c, fv in fill_values.items():
        train_df[c] = train_df[c].fillna(fv)
        if val_df is not None:
            val_df[c] = val_df[c].fillna(fv)
        if test_df is not None:
            test_df[c] = test_df[c].fillna(fv)

    # Scaling FIT on train
    scaler_X = StandardScaler()
    train_df[feature_cols_final] = scaler_X.fit_transform(train_df[feature_cols_final])
    if val_df is not None:
        val_df[feature_cols_final]  = scaler_X.transform(val_df[feature_cols_final])
    if test_df is not None:
        test_df[feature_cols_final] = scaler_X.transform(test_df[feature_cols_final])

    return train_df, val_df, test_df, feature_cols_final, scaler_X

# =============================================================================
# 8. MODEL BUILDER (3 classes)
# =============================================================================
# COST MATRIX 
# 0=DOWN, 1=KEEP, 2=UP

COST_MATRIX = tf.constant([
    [0.0, 1.0, 6.0],  # true DOWN: pred KEEP (1), pred UP (6)  
    [1.0, 0.0, 1.0],  # true KEEP: pred DOWN/UP (1)
    [6.0, 1.0, 0.0],  # true UP  : pred DOWN (6) 
], dtype=tf.float32)

ALPHA_COST = 0.40  

def make_cost_sensitive_loss(cost_matrix, alpha=0.4):
    """
    Loss = (1-alpha)*CE + alpha*ExpectedCost
    - CE = sparse categorical crossentropy
    - ExpectedCost = somma_j cost[y_true, j] * p_pred[j]
    """
    ce_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    def loss(y_true, y_pred):
        # y_true: (batch,) int32/int64
        # y_pred: (batch, 3) probability softmax

        y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)

        # Cross-entropy standard
        ce = ce_fn(y_true_int, y_pred)

        
        row_costs = tf.gather(cost_matrix, y_true_int)

       
        expected_cost = tf.reduce_sum(row_costs * y_pred, axis=1)

       
        return (1.0 - alpha) * ce + alpha * expected_cost

    return loss

def build_lstm_classifier(n_features, cfg):
    model = Sequential([
        Input(shape=(MAX_SEQ_LEN, n_features)),
        Masking(mask_value=PADDING_VALUE),
        LSTM(cfg["lstm_units"], activation='tanh'),
        Dropout(cfg["dropout"]),
        Dense(cfg["dense_units"], activation='relu'),
        Dropout(cfg.get("dropout2", 0.0)),
        Dense(3, activation='softmax')
    ])

    cost_loss = make_cost_sensitive_loss(COST_MATRIX, alpha=ALPHA_COST)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["lr"]),
        loss=cost_loss,
        metrics=['accuracy']
    )
    return model

# =============================================================================
# 9. METRIC + danger errors
# =============================================================================
def eval_classification(model, X, y_true):
    proba = model.predict(X, verbose=0)
    y_pred = np.argmax(proba, axis=1)

    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])

    danger = int(cm[0,2] + cm[2,0])  # DOWN->UP + UP->DOWN

    return {
        "acc": float(acc),
        "bal_acc": float(bal),
        "macro_f1": float(macro),
        "danger": danger,
        "cm": cm,
        "y_pred": y_pred
    }

def pick_cv_score(out, metric):
    if metric == "macro_f1":
        return -out["macro_f1"] 
    if metric == "bal_acc":
        return -out["bal_acc"]
    if metric == "danger":
        return out["danger"]
    raise ValueError("SELECT_METRIC not supported")

# =============================================================================
# 10. SPLIT TRAIN/TEST
# =============================================================================
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
tr_idx, te_idx = next(gss.split(df_model, groups=df_model[ID_COL]))

train_all = df_model.iloc[tr_idx].copy()
test_all  = df_model.iloc[te_idx].copy()

# =============================================================================
# 11. CV
# =============================================================================
lstm_param_list = [
    {"lstm_units": 64, "dense_units": 32, "dropout": 0.30, "dropout2": 0.00, "lr": 5e-4,
     "batch_size": 32, "epochs": 140},
    {"lstm_units": 96, "dense_units": 48, "dropout": 0.35, "dropout2": 0.00, "lr": 5e-4,
     "batch_size": 32, "epochs": 170},
    {"lstm_units": 64, "dense_units": 32, "dropout": 0.40, "dropout2": 0.10, "lr": 3e-4,
     "batch_size": 32, "epochs": 190},
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

    
        train_p, valint_p, valouter_p, feat_final, scaler_X = preprocess_fit_transform(
            train_df, val_int, fold_val_outer, FEATURE_COLS
        )

        X_train, y_train = create_sequences_with_labels(train_p, feat_final, 'y_class', MAX_SEQ_LEN, PADDING_VALUE)
        X_valint, y_valint = create_sequences_with_labels(valint_p, feat_final, 'y_class', MAX_SEQ_LEN, PADDING_VALUE)
        X_valo, y_valo = create_sequences_with_labels(valouter_p, feat_final, 'y_class', MAX_SEQ_LEN, PADDING_VALUE)

        # class weights
        classes = np.array([0,1,2])
        cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

        model = build_lstm_classifier(n_features=len(feat_final), cfg=cfg)

        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(patience=5, factor=0.5, monitor='val_loss')
        ]

        model.fit(
            X_train, y_train,
            validation_data=(X_valint, y_valint),
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=0
        )

        out = eval_classification(model, X_valo, y_valo)
        fold_scores.append(pick_cv_score(out, SELECT_METRIC))

        tf.keras.backend.clear_session()

    mean_score = float(np.mean(fold_scores))

    if SELECT_METRIC in ["macro_f1", "bal_acc"]:
        pretty = -mean_score
        print(f"Config {i}/{len(lstm_param_list)} | {SELECT_METRIC}={pretty:.3f} | units={cfg['lstm_units']}, drop={cfg['dropout']}, lr={cfg['lr']}")
    else:
        print(f"Config {i}/{len(lstm_param_list)} | {SELECT_METRIC}={mean_score:.0f} | units={cfg['lstm_units']}, drop={cfg['dropout']}, lr={cfg['lr']}")

    if mean_score < best_cv_score:
        best_cv_score = mean_score
        best_cfg = cfg

print("\n==================== BEST (da CV su TRAIN) ====================")
if SELECT_METRIC in ["macro_f1", "bal_acc"]:
    print(f"Best {SELECT_METRIC}: {-best_cv_score:.3f}")
else:
    print(f"Best {SELECT_METRIC}: {best_cv_score:.0f}")
print("Best cfg:", best_cfg)

# =============================================================================
# 12. FIT with best_cfg (val interna)
# =============================================================================
gss_val_final = GroupShuffleSplit(n_splits=1, test_size=VAL_SPLIT_IN_TRAIN, random_state=RANDOM_STATE)
trf_idx, vaf_idx = next(gss_val_final.split(train_all, groups=train_all[ID_COL]))

train_df_final = train_all.iloc[trf_idx].copy()
val_int_final  = train_all.iloc[vaf_idx].copy()

train_p, val_p, test_p, feat_final, scaler_X = preprocess_fit_transform(
    train_df_final, val_int_final, test_all, FEATURE_COLS
)

X_train, y_train = create_sequences_with_labels(train_p, feat_final, 'y_class', MAX_SEQ_LEN, PADDING_VALUE)
X_val, y_val     = create_sequences_with_labels(val_p,   feat_final, 'y_class', MAX_SEQ_LEN, PADDING_VALUE)
X_test, y_test   = create_sequences_with_labels(test_p,  feat_final, 'y_class', MAX_SEQ_LEN, PADDING_VALUE)

print("\n==================== FINAL TRAIN ====================")
print("Sequences TRAIN:", X_train.shape, "| dist:", np.bincount(y_train, minlength=3))
print("Sequences VAL  :", X_val.shape,   "| dist:", np.bincount(y_val, minlength=3))
print("Sequences TEST :", X_test.shape,  "| dist:", np.bincount(y_test, minlength=3))
print("Features :", len(feat_final))

# class weights
classes = np.array([0,1,2])
cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

final_model = build_lstm_classifier(n_features=len(feat_final), cfg=best_cfg)

callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
    ReduceLROnPlateau(patience=5, factor=0.5, monitor='val_loss')
]

final_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=best_cfg["epochs"],
    batch_size=best_cfg["batch_size"],
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=0
)

# =============================================================================
# 13. Final Evaluation
# =============================================================================
out_train = eval_classification(final_model, X_train, y_train)
out_test  = eval_classification(final_model, X_test,  y_test)

def pretty_print(name, out, y_true, y_pred):
    print(f"\n==================== {name} ====================")
    print(f"Accuracy:      {out['acc']:.3f}")
    print(f"Balanced Acc:  {out['bal_acc']:.3f}")
    print(f"Macro-F1:      {out['macro_f1']:.3f}")
    print(f"Danger DOWN<->UP: {out['danger']}")
    print("\nConfusion matrix (rows=true, cols=pred) [0=DOWN,1=KEEP,2=UP]:")
    print(out["cm"])
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=[0,1,2], target_names=["DOWN","KEEP","UP"], digits=3))

pretty_print("FINAL TRAIN", out_train, y_train, out_train["y_pred"])
pretty_print("FINAL TEST",  out_test,  y_test,  out_test["y_pred"])


# =============================================================================
# SAVING
# =============================================================================
import os, joblib

SAVE_DIR = "/content/drive/MyDrive/Colab Notebooks/modelli_moe_lstm/"
os.makedirs(SAVE_DIR, exist_ok=True)

final_model.save(os.path.join(SAVE_DIR, "lstm_classifier.keras"))


joblib.dump(scaler_X, os.path.join(SAVE_DIR, "lstm_scaler.pkl"))

joblib.dump(feat_final, os.path.join(SAVE_DIR, "lstm_features.pkl"))

lstm_medians = {c: train_df_final[c].median() for c in FEATURE_COLS}
joblib.dump(lstm_medians, os.path.join(SAVE_DIR, "lstm_medians.pkl"))


meta = {
    "feature_cols": feat_final, 
    "base_feature_cols": FEATURE_COLS,   
    "seq_len": MAX_SEQ_LEN,
    "MAX_SEQ_LEN": MAX_SEQ_LEN,
    "PADDING_VALUE": PADDING_VALUE,
    "TH_REL": TH_REL,
    "TH_UI": TH_UI,
    "isna_cols": [c for c in [
        'Ferritina', 'T-SAT', 'Transferrina', 'Ferritin_TSAT_ratio',
        'Creatinina', 'Vitamina B12',
        'Hb_std_3', 'Hb_Volatility', 'Hb_Trend_5', 'Dose_var_3'
    ] if c in FEATURE_COLS],
}
joblib.dump(meta, os.path.join(SAVE_DIR, "lstm_meta.pkl"))

print("LSTM SAVED succesfully")


