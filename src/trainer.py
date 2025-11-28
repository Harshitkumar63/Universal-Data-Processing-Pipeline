import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from .utils import ensure_dir
from .config import BASE_OUTPUT, TEST_SIZE, RANDOM_STATE

def cross_validate_model(pipeline, X, y, task, cv=5):
    """
    Run cross-validation and return mean score. For regression -> neg MSE -> RMSE reported.
    For classification -> weighted F1.
    """
    if task == "regression":
        scoring = "neg_mean_squared_error"
        kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(pipeline, X, y, scoring=scoring, cv=kf, n_jobs=-1)
        # convert neg_mse to rmse
        rmse = np.sqrt(-scores).mean()
        return rmse
    else:
        scoring = "f1_weighted"
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(pipeline, X, y, scoring=scoring, cv=kf, n_jobs=-1)
        return scores.mean()

class Trainer:
    def __init__(self, df, features, target, task, preprocessor, output_dir=None):
        self.df = df
        self.features = features
        self.target = target
        self.task = task
        self.preprocessor = preprocessor
        self.output_dir = output_dir or BASE_OUTPUT
        ensure_dir(self.output_dir)

    def train_and_select(self, models_dict):
        """
        For each candidate model:
            - build pipeline: preprocessor + model
            - cross-validate on train portion
        choose best model (by lower RMSE for regression or higher f1 for classification).
        Then fit best model on full training data and save model & artifacts.
        Returns metadata dict.
        """
        # split once (train/test)
        X = self.df[self.features]
        y = self.df[self.target]
        stratify = y if self.task == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify)

        results = {}
        for name, model in models_dict.items():
            pipeline = Pipeline([("pre", self.preprocessor), ("model", model)])
            try:
                score = cross_validate_model(pipeline, X_train, y_train, self.task, cv=5)
            except Exception as e:
                score = None
            results[name] = {"score": score}

        # pick best
        if self.task == "regression":
            # lower better (RMSE)
            best_name = min((n for n in results.keys() if results[n]["score"] is not None), key=lambda x: results[x]["score"])
        else:
            # higher better
            best_name = max((n for n in results.keys() if results[n]["score"] is not None), key=lambda x: results[x]["score"])

        best_model = models_dict[best_name]
        best_pipeline = Pipeline([("pre", self.preprocessor), ("model", best_model)])
        best_pipeline.fit(X_train, y_train)

        # save model
        model_path = os.path.join(self.output_dir, f"{best_name}_pipeline.joblib")
        joblib.dump(best_pipeline, model_path)

        # save test split for evaluation
        test_df = X_test.copy()
        test_df[self.target] = y_test
        test_df.to_csv(os.path.join(self.output_dir, "test_split.csv"), index=False)

        return {
            "best_model_name": best_name,
            "model_path": model_path,
            "cv_results": results,
            "test_split_path": os.path.join(self.output_dir, "test_split.csv")
        }
