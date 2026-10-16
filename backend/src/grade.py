import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from features import TextEncoder, build_features, tfidf_cosine, build_features_v3
from preprocessing import clean_text

BACKEND_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BACKEND_DIR / "models" / "grader_models.joblib"
MISSING_POINT_THRESHOLD = 0.5  

def label_feedback(label):
    """Gives feedback based on the predicted label."""
    if label >= 2:
        return "The answer covers all core concepts present in the reference."
    elif label == 1:
        return "The answer is partially correct but misses some key concepts."
    return "The answer is incorrect or fails to address the required concepts."


class V2_NLP_Pipeline:
    """Version 2 (Week 4): hybrid features + tuned Logistic Regression."""

    def __init__(self, bundle, encoder):
        self.bundle = bundle
        self.encoder = encoder

    def _result(self, question_id, label, proba, max_marks, model_name):
        raw_score = (int(label) / 2.0) * max_marks
        return {
            "question": question_id,
            "score": round((int(label) / 2.0) * max_marks),
            "maximum_score": max_marks,
            "feedback": label_feedback(label),
            "confidence": round(float(np.max(proba)), 2),
            "model": model_name,
        }

    def grade_answer_dict(self, question_id, student_answer, reference_answer, max_marks=10):
        """Same as grade_answer(), but returns a dict instead of a JSON string (used by the API)."""
        X, details = build_features(
            [student_answer], [reference_answer],
            self.bundle["tfidf_vectorizer"], self.encoder, return_details=True,
        )
        model = self.bundle["v2_model"]
        label = model.predict(X)[0]
        result = self._result(question_id, label, model.predict_proba(X)[0], max_marks, "v2_hybrid")

        missing = [p for p, sim in details[0] if sim < MISSING_POINT_THRESHOLD]
        result["missing_key_points"] = missing[:3]
        if missing:
            if label >= 2:
                result["feedback"] = (
                    "The answer is graded correct overall, but these reference points were not found: "
                    + "; ".join(missing[:3]) + "."
                )
            else:
                result["feedback"] += " Check these points: " + "; ".join(missing[:3]) + "."
        return result
    
    def grade_answer(self, question_id, student_answer, reference_answer, max_marks=10):
        return json.dumps(self.grade_answer_dict(question_id, student_answer, reference_answer, max_marks), indent=4)

    def grade_with_v3_dict(self, question_id, student_answer, reference_answer, max_marks=10):
        """
        Week 6 default: V3 (Week 6 prep improvement), same hybrid-feature +
        missing-key-point explainability as V2, but using build_features_v3
        (keypoint_min_coverage + negation_mismatch) and the better-performing
        v3_model. Falls back to V2 if a bundle without v3_model is loaded.
        """
        if "v3_model" not in self.bundle:
            return self.grade_answer_dict(question_id, student_answer, reference_answer, max_marks)

        X, details = build_features_v3(
            [student_answer], [reference_answer],
            self.bundle["tfidf_vectorizer"], self.encoder, return_details=True,
        )
        model = self.bundle["v3_model"]
        label = model.predict(X)[0]
        result = self._result(question_id, label, model.predict_proba(X)[0], max_marks, "v3_improved")

        missing = [p for p, sim in details[0] if sim < MISSING_POINT_THRESHOLD]
        result["missing_key_points"] = missing[:3]
        if missing:
            if label >= 2:
                result["feedback"] = (
                    "The answer is graded correct overall, but these reference points were not found: "
                    + "; ".join(missing[:3]) + "."
                )
            else:
                result["feedback"] += " Check these points: " + "; ".join(missing[:3]) + "."
        return result

    def grade_with_v3(self, question_id, student_answer, reference_answer, max_marks=10):
        return json.dumps(self.grade_with_v3_dict(question_id, student_answer, reference_answer, max_marks), indent=4)
    
    def grade_with_baseline_dict(self, question_id, student_answer, reference_answer, max_marks=10):
        s, r = [clean_text(student_answer)], [clean_text(reference_answer)]
        x = tfidf_cosine(self.bundle["tfidf_vectorizer"], s, r).reshape(-1, 1)
        clf = self.bundle["baseline_clf"]
        return self._result(question_id, clf.predict(x)[0], clf.predict_proba(x)[0], max_marks, "baseline_tfidf")

    def grade_with_baseline(self, question_id, student_answer, reference_answer, max_marks=10):
        return json.dumps(self.grade_with_baseline_dict(question_id, student_answer, reference_answer, max_marks), indent=4)

    def grade_with_v1(self, question_id, student_answer, reference_answer, max_marks=10):
        s = self.encoder.encode([clean_text(student_answer)])
        r = self.encoder.encode([clean_text(reference_answer)])
        x = np.sum(s * r, axis=1).reshape(-1, 1)
        clf = self.bundle["v1_clf"]
        return json.dumps(self._result(question_id, clf.predict(x)[0], clf.predict_proba(x)[0],
                                       max_marks, "v1_minilm"), indent=4)


def main():
    parser = argparse.ArgumentParser(description="Grade one answer with Baseline, V1 and V2.")
    parser.add_argument("--question", default="Q4 What does supervised learning use to train model?")
    parser.add_argument("--reference", default=(
        "A supervised machine learning algorithm relies on labeled training data to learn "
        "a function that maps inputs to outputs."))
    parser.add_argument("--answer", default="Supervised learning uses labeled data to train models to predict outcomes.")
    parser.add_argument("--max-marks", type=int, default=10)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(f"{MODEL_PATH} not found. Run: uv run python src/compare_models.py")

    bundle = joblib.load(MODEL_PATH)
    pipeline = V2_NLP_Pipeline(bundle, TextEncoder(bundle["embedding_model"]))
    a = (args.question, args.answer, args.reference, args.max_marks)

    print("=== Baseline (Week 2) ===")
    print(pipeline.grade_with_baseline(*a))
    print("\n=== V1 (Week 3) ===")
    print(pipeline.grade_with_v1(*a))
    print("\n=== V2 (Week 4) ===")
    print(pipeline.grade_answer(*a))
    print("\n=== V3 (Week 6 prep, default) ===")
    print(pipeline.grade_with_v3(*a))

if __name__ == "__main__":
    main()
