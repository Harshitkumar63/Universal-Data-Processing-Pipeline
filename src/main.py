"""
Main entrypoint for the UDPP project.

Usage:
    python -m src.main --data data/regression.csv --target price
"""
import argparse 
import os
from .utils import read_data, ensure_dir
from .preprocess import preprocess_dataframe, build_preprocessor, detect_column_types
from .models import get_model_candidates
from .trainer import Trainer
from .evaluate import evaluate_model
from .config import BASE_OUTPUT

def main(data, target=None, output=None):
    output = output or BASE_OUTPUT
    ensure_dir(output) 

    # 1) load
    df, name = read_data(data)
    print(f"[INFO] Loaded {name} shape={df.shape}") 

    # 2) detect target
    if target is None:
        # choose last column (simple fallback)
        target = df.columns[-1]
        print(f"[WARN] No --target passed. Using last column: {target}")
    else:
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataset columns: {list(df.columns)}")

    # 3) basic type detection and preprocessing
    types = detect_column_types(df.drop(columns=[target]))
    datetime_cols = types.get("datetime", None)
    processed_df, numeric, categorical = preprocess_dataframe(df, datetime_cols=datetime_cols)
    features = numeric + categorical
    print(f"[INFO] Features count: {len(features)} (numeric={len(numeric)}, categorical={len(categorical)})")

    # 4) build preprocessor
    preprocessor = build_preprocessor(processed_df, numeric, categorical)

    # 5) detect task
    target_ser = processed_df[target]
    if target_ser.dtype == 'object' or target_ser.nunique() <= 20:
        task = "classification"
    else:
        task = "regression"
    print(f"[INFO] Detected task: {task}")

    # 6) get model candidates
    models = get_model_candidates(task=task)

    # 7) train & select
    dataset_output = os.path.join(output, name)
    ensure_dir(dataset_output)
    trainer = Trainer(processed_df, features, target, task, preprocessor, output_dir=dataset_output)
    result = trainer.train_and_select(models)
    print(f"[INFO] Training finished. Best model: {result['best_model_name']}")

    # 8) final evaluation on saved test split (load it)
    test_split = pd.read_csv(result['test_split_path'])
    X_test = test_split[features]
    y_test = test_split[target]
    # load saved pipeline
    import joblib
    pipeline = joblib.load(result['model_path'])
    metrics = evaluate_model(pipeline, X_test, y_test, task, dataset_output)
    print("[INFO] Evaluation metrics:", metrics)
    print("[INFO] All outputs saved at:", dataset_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UDPP - Universal Data Processing Pipeline")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV or Excel dataset")
    parser.add_argument("--target", type=str, default=None, help="Target column name (optional)")
    parser.add_argument("--output", type=str, default=None, help="Output folder (optional)")
    args = parser.parse_args()
    main(args.data, target=args.target, output=args.output)
