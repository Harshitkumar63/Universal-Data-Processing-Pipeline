import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_squared_error, r2_score,
                             accuracy_score, f1_score, confusion_matrix, classification_report)
from .utils import ensure_dir

def evaluate_model(pipeline, X_test, y_test, task, output_dir):
    """
    Compute metrics and save plots/images. Returns a metrics dict.
    """
    ensure_dir(output_dir)
    y_pred = pipeline.predict(X_test)
    metrics = {}
    if task == "regression":
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        metrics["rmse"] = float(rmse)
        metrics["r2"] = float(r2)
        # residual plot
        plt.figure()
        plt.scatter(y_pred, y_test - y_pred)
        plt.axhline(0, linestyle="--")
        plt.xlabel("Predicted")
        plt.ylabel("Residual (Actual - Predicted)")
        plt.title("Residuals")
        plt.savefig(os.path.join(output_dir, "residuals.png"), bbox_inches='tight')
        plt.close()
    else:
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        metrics["accuracy"] = float(acc)
        metrics["f1_weighted"] = float(f1)
        # confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure()
        plt.imshow(cm, interpolation='nearest')
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), bbox_inches='tight')
        plt.close()
        # textual report
        report = classification_report(y_test, y_pred, output_dict=True)
        pd.DataFrame(report).to_csv(os.path.join(output_dir, "classification_report.csv"))

    return metrics
