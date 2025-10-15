'''
model from mxmh_AI4ALL.ipynb
Exposes:
- get_ui_schema()
- get_label_map()
- predict_from_user_dict(d: dict)
'''

from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

try:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
except Exception:
    kagglehub = None
    KaggleDatasetAdapter = None

# Globals cached after first load/fit
_MODEL = None
_OE = None
_LE = None
_CAT_COLS: List[str] = []
_FEATURE_COLS: List[str] = []
_TARGET_COL = "Music effects"
_TRAIN_DF: pd.DataFrame | None = None


def load_data_via_kagglehub() -> pd.DataFrame:
    # Load the dataset exactly like in notebook
    if kagglehub is None or KaggleDatasetAdapter is None:
        raise RuntimeError("kagglehub is not available. Install with: pip install kagglehub[pandas-datasets]")
    file_path = "mxmh_survey_results.csv"
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "catherinerasgaitis/mxmh-survey-results",
        file_path,
        # Provide any additional arguments like
        # sql_query or pandas_kwargs. See the
        # documentation for more information:
        # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
    )
    return df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = ["Timestamp", "Primary streaming service", "BPM", "Permissions"]
    df = df.drop(columns = cols_to_drop, errors = 'ignore').copy()
    df.dropna(axis = 0, inplace = True)
    return df


def fit_artifacts(df: pd.DataFrame) -> Tuple[RandomForestClassifier, OrdinalEncoder, LabelEncoder, List[str], List[str]]:
    # Fit encoders and train the final RandomForest on the full cleaned data
    if _TARGET_COL not in df.columns:
        raise KeyError(f"Expected target column '{_TARGET_COL}' not found in data.")
    y = df[_TARGET_COL]
    X = df.drop(columns = [_TARGET_COL])

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]

    # Encoders
    oe = OrdinalEncoder()
    le = LabelEncoder()

    y_enc = le.fit_transform(y)
    X_enc = X.copy()
    if len(cat_cols) > 0:
        X_enc[cat_cols] = oe.fit_transform(X[cat_cols])

    # Final model (as in notebook)
    model = RandomForestClassifier(n_estimators = 100, random_state = 0)
    model.fit(X_enc, y_enc)

    feature_cols = list(X.columns)
    return model, oe, le, cat_cols, feature_cols


def _ensure_fitted() -> None:
    # Fit/cache global artifacts the first time they're needed
    global _MODEL, _OE, _LE, _CAT_COLS, _FEATURE_COLS, _TRAIN_DF
    if _MODEL is None:
        df_raw = load_data_via_kagglehub()
        df = clean_df(df_raw)
        _TRAIN_DF = df.copy()
        _MODEL, _OE, _LE, _CAT_COLS, _FEATURE_COLS = fit_artifacts(df)


def get_ui_schema() -> Dict[str, Any]:
    # Return a schema to build Streamlit inputs
    _ensure_fitted()
    assert _TRAIN_DF is not None
    df = _TRAIN_DF 
    X = df.drop(columns = [_TARGET_COL])

    categorical: Dict[str, list] = {}
    numerical: Dict[str, dict] = {}

    for c in X.columns:
        if c in _CAT_COLS:
            choices = sorted([v for v in X[c].dropna().unique().tolist()])
            categorical[c] = choices
        else:
            s = pd.to_numeric(X[c], errors = "coerce").dropna()
            if len(s) == 0:
                categorical[c] = sorted([v for v in X[c].dropna().unique().tolist()])
            else:
                numerical[c] = {
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "median": float(s.median()),
                }
    return {
        "feature_cols": list(X.columns),
        "categorical": categorical,
        "numerical": numerical,
    }


def get_label_map() -> Dict[int, str]:
    _ensure_fitted()
    return {i: cls for i, cls in enumerate(_LE.classes_)}


def _encode_features(input_row: pd.DataFrame) -> pd.DataFrame:
    _ensure_fitted()
    X = input_row.copy()
    if len(_CAT_COLS) > 0:
        X[_CAT_COLS] = _OE.transform(X[_CAT_COLS])
    return X


def predict_from_user_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    # Predict from a dict of user inputs whose keys match feature column names.
    _ensure_fitted()
    missing = [c for c in _FEATURE_COLS if c not in d]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    
    X_row = pd.DataFrame([{c: d[c] for c in _FEATURE_COLS}])
    X_row_enc = _encode_features(X_row)

    pred_id = int(_MODEL.predict(X_row_enc)[0])
    proba = {}
    if hasattr(_MODEL, "predict_proba"):
        probs = _MODEL.predict_proba(X_row_enc)[0]
        label_map = get_label_map()
        proba = {label_map[i]: float(p) for i, p in enumerate(probs)}

    label = get_label_map()[pred_id]
    return {"pred_label_id": pred_id, "pred_label": label, "proba": proba}

