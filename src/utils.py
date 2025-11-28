import os
import pandas as pd

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def read_data(path):
    """Read CSV or Excel into DataFrame. Returns (df, filename_without_ext)."""
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    fname = os.path.splitext(os.path.basename(path))[0]
    return df, fname
