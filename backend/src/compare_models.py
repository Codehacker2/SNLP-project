"""
Week 4 - Baseline vs Improved model comparison.

Trains and evaluates three models on the SAME EngSAF splits that Week 2 created
(data/engsaf_train.csv, engsaf_validation.csv, engsaf_test.csv):

    Baseline (Week 2): TF-IDF cosine similarity      -> Logistic Regression
    V1       (Week 3): MiniLM cosine similarity      -> Logistic Regression
    V2       (Week 4): 7 hybrid features (features.py)-> scaled, class-balanced
                       Logistic Regression, tuned on the validation set

The Baseline and V1 code reproduces the two notebooks exactly, so their test
accuracy should match what we reported before (0.5020 and 0.5000).

Outputs
    results/model_comparison.md   tables for the report (copy into the PDF)
    results/model_comparison.csv  same numbers as a CSV
    models/grader_models.joblib   trained models, used by src/grade.py

Run from the backend/ folder:
    uv run python src/compare_models.py
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import EMBEDDING_MODEL_NAME, FEATURE_NAMES, TextEncoder, build_features, tfidf_cosine

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"
RESULTS_DIR = BACKEND_DIR / "results"
MODELS_DIR = BACKEND_DIR / "models"

LABELS = [0, 1, 2]  # 0 = incorrect, 1 = partially correct, 2 = correct
TEXT_COLS = ["student_answer", "reference_answer", "student_answer_clean", "reference_answer_clean"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def load_splits():
    splits = {}
    for name in ["train", "validation", "test"]:
        df = pd.read_csv(DATA_DIR / f"engsaf_{name}.csv")
        for col in TEXT_COLS:
            df[col] = df[col].fillna("").astype(str)
        splits[name] = df
    return splits


def evaluate(y_true, y_pred):
    """All metrics used in the comparison table."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "mae": mean_absolute_error(y_true, y_pred),
        "partial_recall": recall_score(y_true, y_pred, labels=[1], average="macro", zero_division=0),
    }


def md_table(headers, rows):
    """Tiny markdown table writer (avoids needing the 'tabulate' package)."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def fmt(x):
    return f"{x:.4f}"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    splits = load_splits()
    train, val, test = splits["train"], splits["validation"], splits["test"]
    print(f"Train {train.shape}, Validation {val.shape}, Test {test.shape}")
    y = {name: df["score"].values for name, df in splits.items()}

    # ---------------- Baseline (Week 2) ----------------
    print("\n[1/3] Baseline: TF-IDF cosine + Logistic Regression (Week 2)")
    t0 = time.time()
    vectorizer = TfidfVectorizer()
    vectorizer.fit(pd.concat([train["student_answer_clean"], train["reference_answer_clean"]]))
    X_base = {
        n: tfidf_cosine(vectorizer, df["student_answer_clean"], df["reference_answer_clean"]).reshape(-1, 1)
        for n, df in splits.items()
    }
    base_clf = LogisticRegression().fit(X_base["train"], y["train"])
    pred_base = {n: base_clf.predict(X_base[n]) for n in splits}
    time_base = time.time() - t0

    # ---------------- V1 (Week 3) ----------------
    print(f"\n[2/3] V1: {EMBEDDING_MODEL_NAME} cosine + Logistic Regression (Week 3)")
    t0 = time.time()
    encoder = TextEncoder(EMBEDDING_MODEL_NAME)
    X_v1 = {}
    for n, df in splits.items():
        s = encoder.encode(df["student_answer_clean"].tolist())
        r = encoder.encode(df["reference_answer_clean"].tolist())
        X_v1[n] = np.sum(s * r, axis=1).reshape(-1, 1)
    v1_clf = LogisticRegression().fit(X_v1["train"], y["train"])
    pred_v1 = {n: v1_clf.predict(X_v1[n]) for n in splits}
    time_v1 = time.time() - t0

    # ---------------- V2 (Week 4) ----------------
    print("\n[3/3] V2: hybrid features + tuned Logistic Regression (Week 4)")
    t0 = time.time()
    X_v2 = {
        n: build_features(df["student_answer"].tolist(), df["reference_answer"].tolist(), vectorizer, encoder)
        for n, df in splits.items()
    }

    # Small, honest hyper-parameter search: choose on VALIDATION macro-F1,
    # the test set is only used once at the end.
    def make_v2(C, cw):
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight=cw, max_iter=2000))

    best = None
    for C in [0.1, 1.0, 10.0]:
        for cw in [None, "balanced"]:
            m = make_v2(C, cw).fit(X_v2["train"], y["train"])
            score = f1_score(y["validation"], m.predict(X_v2["validation"]), average="macro")
            print(f"    C={C:<5} class_weight={str(cw):<9} -> validation macro-F1 = {score:.4f}")
            if best is None or score > best[0]:
                best = (score, C, cw, m)
    _, best_C, best_cw, v2_model = best
    print(f"    Selected: C={best_C}, class_weight={best_cw}")
    pred_v2 = {n: v2_model.predict(X_v2[n]) for n in splits}
    time_v2 = time.time() - t0

    # ---------------- Comparison tables ----------------
    models = [
        ("Baseline - TF-IDF + LR (Week 2)", pred_base, time_base),
        ("V1 - MiniLM + LR (Week 3)", pred_v1, time_v1),
        ("V2 - Hybrid features + LR (Week 4)", pred_v2, time_v2),
    ]
    csv_rows, md = [], ["# Week 4 - Baseline vs Improved Model Comparison", ""]
    md.append("Dataset: EngSAF (train 4733 / validation 338 / test 506). "
              "Labels: 0 = incorrect, 1 = partially correct, 2 = correct.")
    md.append("")
    md.append("Metrics: Accuracy; Macro-F1 (treats the three classes equally); QWK = quadratic "
              "weighted kappa (agreement with the human grade, penalises big mistakes more); "
              "MAE = mean absolute error in label points (lower is better); Partial recall = "
              "share of 'partially correct' answers the model recognised.")

    for split in ["test", "validation"]:
        md += ["", f"## {split.capitalize()} set", ""]
        rows = []
        for name, preds, secs in models:
            m = evaluate(y[split], preds[split])
            rows.append([name, fmt(m["accuracy"]), fmt(m["macro_f1"]), fmt(m["qwk"]),
                         fmt(m["mae"]), fmt(m["partial_recall"])])
            csv_rows.append({"split": split, "model": name, **m, "train_and_predict_seconds": round(secs, 1)})
        md.append(md_table(["Model", "Accuracy", "Macro-F1", "QWK", "MAE", "Partial recall"], rows))

    md += ["", "## Confusion matrices (test set)", "",
           "Rows = true label, columns = predicted label (0 / 1 / 2)."]
    for name, preds, _ in models:
        cm = confusion_matrix(y["test"], preds["test"], labels=LABELS)
        md += ["", f"**{name}**", "",
               md_table(["true \\ pred", "0", "1", "2"], [[LABELS[i], *cm[i]] for i in range(3)])]

    # ---------------- Ablation: which features helped? ----------------
    ablations = [
        ("MiniLM cosine only", [0]),
        ("+ TF-IDF cosine", [0, 1]),
        ("+ word overlap + length features", [0, 1, 3, 4, 5, 6]),
        ("+ key-point coverage (full V2)", [0, 1, 2, 3, 4, 5, 6]),
    ]
    ab_rows = []
    for name, cols in ablations:
        m = make_v2(best_C, best_cw).fit(X_v2["train"][:, cols], y["train"])
        r = evaluate(y["test"], m.predict(X_v2["test"][:, cols]))
        ab_rows.append([name, fmt(r["accuracy"]), fmt(r["macro_f1"]), fmt(r["qwk"])])
    md += ["", "## Ablation (test set, same classifier settings as V2)", "",
           md_table(["Features used", "Accuracy", "Macro-F1", "QWK"], ab_rows)]

    # ---------------- What the V2 model learned ----------------
    lr = v2_model.named_steps["logisticregression"]
    coef_correct = lr.coef_[list(lr.classes_).index(2)]
    md += ["", "## V2 feature weights for the 'correct' class", "",
           "Features are standardised, so weights are comparable. Positive = pushes the "
           "prediction towards 'correct'.", "",
           md_table(["Feature", "Weight"], [[f, f"{w:+.3f}"] for f, w in zip(FEATURE_NAMES, coef_correct)])]
    md += ["", f"V2 hyper-parameters (chosen on validation macro-F1): C={best_C}, class_weight={best_cw}"]

    report = "\n".join(md) + "\n"
    (RESULTS_DIR / "model_comparison.md").write_text(report, encoding="utf-8")
    pd.DataFrame(csv_rows).to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    # ---------------- Save models for src/grade.py ----------------
    joblib.dump(
        {
            "embedding_model": EMBEDDING_MODEL_NAME,
            "feature_names": FEATURE_NAMES,
            "tfidf_vectorizer": vectorizer,
            "baseline_clf": base_clf,
            "v1_clf": v1_clf,
            "v2_model": v2_model,
        },
        MODELS_DIR / "grader_models.joblib",
    )

    print("\n" + report)
    print(f"Saved: {RESULTS_DIR / 'model_comparison.md'}")
    print(f"Saved: {RESULTS_DIR / 'model_comparison.csv'}")
    print(f"Saved: {MODELS_DIR / 'grader_models.joblib'}")


if __name__ == "__main__":
    main()
