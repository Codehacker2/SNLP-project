"""
Week 5-style evaluation + error analysis for the V3 model from
improve_model.py. Mirrors week5.ipynb's structure (metrics, confusion
matrix, error-type breakdown, confidence summary) but against the improved
model, on the same untouched test split.

Run from backend/: `python src/evaluate_v3.py`
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, cohen_kappa_score,
    f1_score, mean_absolute_error, recall_score, classification_report,
)

from compare_models import DATA_DIR, MODELS_DIR, RESULTS_DIR
from features import TextEncoder, build_features_v3


def main():
    test_df = pd.read_csv(DATA_DIR / "engsaf_test.csv")
    bundle = joblib.load(MODELS_DIR / "grader_models.joblib")
    encoder = TextEncoder(bundle["embedding_model"])

    X_test = build_features_v3(
        test_df["student_answer"].tolist(),
        test_df["reference_answer"].tolist(),
        bundle["tfidf_vectorizer"],
        encoder,
    )
    v3_model = bundle["v3_model"]
    y_true = test_df["score"].to_numpy()
    y_pred = v3_model.predict(X_test)
    y_proba = v3_model.predict_proba(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro-F1": f1_score(y_true, y_pred, average="macro"),
        "QWK": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "MAE": mean_absolute_error(y_true, y_pred),
        "Partial Recall": recall_score(y_true, y_pred, labels=[1], average="macro", zero_division=0),
    }
    print("=== V3 Test Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    print("\n=== Confusion Matrix (rows=true, cols=pred) ===")
    print(pd.DataFrame(cm, index=["True:0", "True:1", "True:2"], columns=["Pred:0", "Pred:1", "Pred:2"]))

    print("\n=== Per-class report ===")
    print(classification_report(y_true, y_pred, target_names=["Incorrect", "Partial", "Correct"]))

    error_df = test_df.copy()
    error_df["predicted_score"] = y_pred
    error_df["confidence"] = y_proba.max(axis=1)
    error_df["error_type"] = np.select(
        [
            (error_df["score"] == 0) & (error_df["predicted_score"] == 1),
            (error_df["score"] == 0) & (error_df["predicted_score"] == 2),
            (error_df["score"] == 1) & (error_df["predicted_score"] == 0),
            (error_df["score"] == 1) & (error_df["predicted_score"] == 2),
            (error_df["score"] == 2) & (error_df["predicted_score"] == 0),
            (error_df["score"] == 2) & (error_df["predicted_score"] == 1),
        ],
        [
            "Incorrect -> Partial", "Incorrect -> Correct", "Partial -> Incorrect",
            "Partial -> Correct", "Correct -> Incorrect", "Correct -> Partial",
        ],
        default="Correct prediction",
    )
    errors_only = error_df[error_df["score"] != error_df["predicted_score"]]
    print(f"\nTotal errors: {len(errors_only)} / {len(error_df)} ({len(errors_only)/len(error_df):.4f})")
    print("\n=== Error type breakdown ===")
    print(errors_only["error_type"].value_counts())

    error_df["correct_prediction"] = error_df["score"] == error_df["predicted_score"]
    conf_summary = error_df.groupby("correct_prediction")["confidence"].agg(["count", "mean", "median", "min", "max"]).round(4)
    print("\n=== Confidence by correctness ===")
    print(conf_summary)

    # Save the per-row results so they can be inspected/compared to Week 5's
    # failure-case table without re-running the model.
    out_path = RESULTS_DIR / "v3_test_predictions.csv"
    error_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
