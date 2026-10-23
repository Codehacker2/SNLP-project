"""Contract checks for the Group 1 booklet bridge, without loading ML models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from booklet import GradeBookletRequest, grade_booklet


def test_booklet_grades_only_eligible_answers():
    request = GradeBookletRequest.model_validate({
        "booklet": {
            "schema_version": "1.0", "booklet_id": "B1", "student_id": "S1",
            "answers": [
                {"question": "Q1", "answer_text": "corrected text", "page": 1,
                 "confidence": 0.8, "needs_review": True, "review": {"status": "corrected"}},
                {"question": "Q2", "answer_text": "unclear", "page": 1,
                 "confidence": 0.2, "needs_review": True, "flags": ["low_confidence"]},
                {"question": "Q3", "answer_text": "unsafe", "page": 2,
                 "confidence": 0.9, "needs_review": False},
            ],
        },
        "answer_key": {
            "Q1": {"question_text": "What is Q1?", "reference_answer": "reference",
                   "rubric": ["point A", "point B"], "max_marks": 5},
            "Q2": {"reference_answer": "reference"},
            "Q3": {"reference_answer": "reference"},
            "Q4": {"reference_answer": "reference"},
        },
    })
    calls = []

    def evaluate(question, reference, rubric, answer, marks):
        calls.append((question, reference, rubric, answer, marks))
        if answer == "unsafe":
            return {"grading_status": "human_review", "score": None,
                    "promptguard": {"recommended_action": "BLOCK"}}
        return {"grading_status": "graded", "score": 4, "maximum_score": 5,
                "feedback": "Good", "confidence": 0.75, "model": "v3_improved"}

    result = grade_booklet(request, evaluate)
    assert calls == [
        ("What is Q1?", "reference", "point A; point B", "corrected text", 5),
        ("Q3", "reference", "", "unsafe", 10),
    ]
    assert result["auto_marks"] == 4 and result["auto_max"] == 5
    assert result["graded"][0]["confidence"] == 0.6
    assert result["graded"][0]["human_reviewed"] is True
    assert [item["question"] for item in result["review_queue"]] == ["Q2", "Q3"]
    assert result["questions_not_found"] == ["Q4"]


def test_unknown_student_never_reaches_grader():
    request = GradeBookletRequest.model_validate({
        "booklet": {"schema_version": "1.0", "booklet_id": "B2", "student_id": None,
                    "answers": [{"question": "Q1", "answer_text": "text", "page": 1,
                                 "confidence": 1, "needs_review": False}]},
        "answer_key": {"Q1": {"reference_answer": "reference"}},
    })
    result = grade_booklet(request, lambda *args: (_ for _ in ()).throw(AssertionError("graded")))
    assert result["graded"] == []
    assert result["review_queue"][0]["reason"] == "student not identified"
