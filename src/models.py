from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# XGBoost/LightGBM can be added optionally
try:
    from xgboost import XGBRegressor, XGBClassifier
except Exception:
    XGBRegressor = XGBClassifier = None

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
except Exception:
    LGBMRegressor = LGBMClassifier = None

def get_model_candidates(task="regression"):
    """
    Returns dict of model_name -> model_instance (basic candidates).
    Add or remove as needed.
    """
    if task == "regression":
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
        }
        if XGBRegressor is not None:
            models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        if LGBMRegressor is not None:
            models["LightGBM"] = LGBMRegressor(n_estimators=100, random_state=42)
    else:
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
        }
        if XGBClassifier is not None:
            models["XGBoost"] = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
        if LGBMClassifier is not None:
            models["LightGBM"] = LGBMClassifier(n_estimators=100, random_state=42)
    return models
