import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from compare_models import DATA_DIR, MODELS_DIR, RESULTS_DIR, evaluate, fmt, load_splits, md_table
from features import EMBEDDING_MODEL_NAME, TextEncoder, V3_FEATURE_NAMES, build_features_v3

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    splits = load_splits()
    train, val, test = splits["train"], splits["validation"], splits["test"]
    y = {name: df["score"].values for name, df in splits.items()}
    print(f"Train {train.shape}, Validation {val.shape}, Test {test.shape}")

    print("\nFitting TF-IDF vectorizer (same recipe as Week 2/4)...")
    vectorizer = TfidfVectorizer()
    vectorizer.fit(pd.concat([train["student_answer_clean"], train["reference_answer_clean"]]))

    print("Loading MiniLM encoder (reused for every row, cached)...")
    encoder = TextEncoder(EMBEDDING_MODEL_NAME)

    print("\nBuilding V3 features for train/validation/test...")
    t0 = time.time()
    X = {
        n: build_features_v3(df["student_answer"].tolist(), df["reference_answer"].tolist(), vectorizer, encoder)
        for n, df in splits.items()
    }
    build_time = time.time() - t0
    print(f"  done in {build_time:.1f}s")

    # Honest model search: every candidate is scored on VALIDATION macro-F1.
    # The test set isn't touched until the single best candidate is picked.
    candidates = []
    for C in [0.1, 1.0, 10.0]:
        for cw in [None, "balanced"]:
            candidates.append((
                f"LogisticRegression(C={C}, class_weight={cw})",
                make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight=cw, max_iter=2000)),
            ))
    for n_estimators in [200, 400]:
        for max_depth in [None, 8]:
            candidates.append((
                f"RandomForest(n_estimators={n_estimators}, max_depth={max_depth})",
                RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0, class_weight="balanced"),
            ))
    for n_estimators in [150, 300]:
        for lr in [0.05, 0.1]:
            candidates.append((
                f"GradientBoosting(n_estimators={n_estimators}, lr={lr})",
                GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=lr, max_depth=3, random_state=0),
            ))

    print("\nSearching classifiers (selected on validation macro-F1)...")
    best = None
    search_rows = []
    for name, model in candidates:
        model.fit(X["train"], y["train"])
        val_pred = model.predict(X["validation"])
        score = f1_score(y["validation"], val_pred, average="macro")
        search_rows.append((name, score))
        print(f"    {name:<55} -> validation macro-F1 = {score:.4f}")
        if best is None or score > best[0]:
            best = (score, name, model)

    best_score, best_name, best_model = best
    print(f"\nSelected: {best_name} (validation macro-F1 = {best_score:.4f})")

    # Test set touched once, here.
    pred_test = best_model.predict(X["test"])
    pred_val = best_model.predict(X["validation"])
    metrics_test = evaluate(y["test"], pred_test)
    metrics_val = evaluate(y["validation"], pred_val)

    # Also report the old V2 model on the same test set for a direct comparison.
    bundle_path = MODELS_DIR / "grader_models.joblib"
    v2_comparison_rows = []
    if bundle_path.exists():
        old_bundle = joblib.load(bundle_path)
        from features import build_features  # V2's original feature fn
        X_v2_test = build_features(test["student_answer"].tolist(), test["reference_answer"].tolist(),
                                    old_bundle["tfidf_vectorizer"], encoder)
        v2_pred_test = old_bundle["v2_model"].predict(X_v2_test)
        v2_metrics = evaluate(y["test"], v2_pred_test)
        v2_comparison_rows.append(["V2 - Hybrid features + LR (Week 4)", fmt(v2_metrics["accuracy"]),
                                    fmt(v2_metrics["macro_f1"]), fmt(v2_metrics["qwk"]),
                                    fmt(v2_metrics["mae"]), fmt(v2_metrics["partial_recall"])])

    md = ["# Week 6 prep - V3 accuracy-improvement pass", ""]
    md.append("Same EngSAF splits as Week 2-5 (train 4733 / validation 338 / test 506), untouched. "
              "V3 keeps every Week 4 V2 feature except `keypoint_coverage` (replaced by "
              "`keypoint_min_coverage`) and adds `negation_mismatch`. See `features.py` for why.")
    md.append("")
    md.append("## Classifier search (validation macro-F1)")
    md.append("")
    md.append(md_table(["Candidate", "Validation macro-F1"],
                        [[n, f"{s:.4f}"] for n, s in search_rows]))
    md.append("")
    md.append(f"**Selected: {best_name}**")
    md.append("")
    md.append("## Test set: V2 (Week 4) vs V3 (this pass)")
    md.append("")
    rows = list(v2_comparison_rows)
    rows.append([f"V3 - {best_name}", fmt(metrics_test["accuracy"]), fmt(metrics_test["macro_f1"]),
                 fmt(metrics_test["qwk"]), fmt(metrics_test["mae"]), fmt(metrics_test["partial_recall"])])
    md.append(md_table(["Model", "Accuracy", "Macro-F1", "QWK", "MAE", "Partial recall"], rows))
    md.append("")
    md.append("## Validation set (V3)")
    md.append("")
    md.append(md_table(["Metric", "Value"], [[k, f"{v:.4f}"] for k, v in metrics_val.items()]))

    report = "\n".join(md) + "\n"
    (RESULTS_DIR / "model_comparison_v3.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Saved: {RESULTS_DIR / 'model_comparison_v3.md'}")

    # Extend (not replace) the existing bundle so grade.py can offer V3
    # alongside baseline/V1/V2 without breaking anything that already works.
    bundle = joblib.load(bundle_path) if bundle_path.exists() else {}
    bundle.update({
        "embedding_model": EMBEDDING_MODEL_NAME,
        "v3_feature_names": V3_FEATURE_NAMES,
        "v3_model": best_model,
        "v3_model_name": best_name,
    })
    joblib.dump(bundle, bundle_path)
    print(f"Saved: {bundle_path} (added v3_model)")


if __name__ == "__main__":
    main()
