import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from collections import Counter
from .config import TOP_CATEGORY_K

def detect_column_types(df):
    """
    Identify numeric, categorical, datetime columns.
    Return dict with lists: numeric, categorical, datetime.
    """
    numeric = df.select_dtypes(include=['number']).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    datetime = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
    return {"numeric": numeric, "categorical": categorical, "datetime": datetime}

def reduce_cardinality(series, top_k=TOP_CATEGORY_K):
    """
    Keep top_k categories and mark others as '__OTHER__' to limit one-hot explosion.
    """
    top = [v for v, _ in Counter(series.fillna("__NA__")).most_common(top_k)]
    return series.fillna("__NA__").apply(lambda x: x if x in top else "__OTHER__")

def build_preprocessor(df, numeric_columns, categorical_columns):
    """
    Build sklearn ColumnTransformer that imputes/scales numerics and imputes/one-hot-encodes categoricals.
    Returns the transformer (not yet fitted).
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__NA__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, categorical_columns)
        ],
        remainder="drop"  # drop any columns not specified
    )
    return preprocessor

def preprocess_dataframe(df, datetime_cols=None, top_k=TOP_CATEGORY_K):
    """
    Basic preprocessing that:
    - expands datetime cols into year/month/day/weekday if present,
    - reduces cardinality on categorical features,
    - returns processed dataframe and lists of numeric/categorical names.
    """
    df = df.copy()
    if datetime_cols:
        for c in datetime_cols:
            df[c] = pd.to_datetime(df[c], errors='coerce')
            df[f"{c}_year"] = df[c].dt.year
            df[f"{c}_month"] = df[c].dt.month
            df[f"{c}_day"] = df[c].dt.day
            df[f"{c}_weekday"] = df[c].dt.weekday
    # detect types again after expansion
    types = detect_column_types(df)
    for c in types["categorical"]:
        df[c] = reduce_cardinality(df[c], top_k=top_k)

    # final lists
    numeric = types["numeric"] + [col for col in df.columns if col.endswith(("_year","_month","_day","_weekday"))]
    categorical = [c for c in types["categorical"] if c in df.columns]

    # ensure no duplicates and maintain order
    numeric = [c for i, c in enumerate(numeric) if c in df.columns and c not in numeric[:i]]
    categorical = [c for i, c in enumerate(categorical) if c in df.columns and c not in categorical[:i]]

    return df, numeric, categorical
