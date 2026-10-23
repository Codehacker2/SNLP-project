"""Bridge the Group 1 digitizer's schema 1.0 to the Group 2 grader."""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field


class AnswerReview(BaseModel):
    status: str | None = None


class DigitizedAnswer(BaseModel):
    question: str
    answer_text: str
    page: int = Field(ge=1)
    confidence: float = Field(ge=0, le=1)
    needs_review: bool
    flags: list[str] = Field(default_factory=list)
    review: AnswerReview | None = None


class DigitizedBooklet(BaseModel):
    schema_version: str
    booklet_id: str
    student_id: str | None = None
    answers: list[DigitizedAnswer]


class AnswerKeyEntry(BaseModel):
    reference_answer: str
    question_text: str = ""
    rubric: list[str] = Field(default_factory=list)
    max_marks: int = Field(default=10, ge=1)


class GradeBookletRequest(BaseModel):
    booklet: DigitizedBooklet
    answer_key: dict[str, AnswerKeyEntry]


def grade_booklet(
    request: GradeBookletRequest,
    evaluate: Callable[[str, str, str, str, int], dict[str, Any]],
) -> dict[str, Any]:
    """Apply review gates, then grade eligible answers through the real API flow.

    ``evaluate`` receives question, reference, rubric, student answer and marks.
    It must include the grader's PromptGuard step.
    """
    booklet = request.booklet
    if booklet.schema_version != "1.0":
        raise ValueError("Only digitizer schema_version 1.0 is supported.")

    graded: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []
    seen: set[str] = set()
    for answer in booklet.answers:
        item = {
            "question": answer.question,
            "page": answer.page,
            "digitizer_confidence": answer.confidence,
        }
        approved = answer.review is not None and answer.review.status in {"approved", "corrected"}
        if not booklet.student_id:
            reason = "student not identified"
        elif answer.question in seen:
            reason = "duplicate question in booklet"
        elif answer.needs_review and not approved:
            reason = "digitizer: " + (", ".join(answer.flags) or "low confidence")
        elif not answer.answer_text.strip():
            reason = "empty answer transcript"
        elif answer.question not in request.answer_key:
            reason = "question not in the answer key"
        else:
            reason = None
        seen.add(answer.question)
        if reason:
            review_queue.append({**item, "reason": reason})
            continue

        key = request.answer_key[answer.question]
        result = evaluate(
            key.question_text or answer.question,
            key.reference_answer,
            "; ".join(key.rubric),
            answer.answer_text,
            key.max_marks,
        )
        if result.get("grading_status") == "human_review" or result.get("score") is None:
            review_queue.append({
                **item,
                "reason": "PromptGuard: human review required",
                "promptguard": result.get("promptguard", {}),
            })
            continue

        grader_confidence = result.get("confidence")
        graded.append({
            **item,
            "score": result["score"],
            "maximum_score": result["maximum_score"],
            "feedback": result.get("feedback"),
            "missing_key_points": result.get("missing_key_points", []),
            "model": result.get("model"),
            "promptguard": result.get("promptguard", {}),
            "grader_confidence": grader_confidence,
            "confidence": round(answer.confidence * grader_confidence, 3) if grader_confidence is not None else None,
            "human_reviewed": bool(approved),
        })

    return {
        "booklet_id": booklet.booklet_id,
        "student_id": booklet.student_id,
        "graded": graded,
        "review_queue": review_queue,
        "questions_not_found": sorted(set(request.answer_key) - seen),
        "auto_marks": sum(entry["score"] for entry in graded),
        "auto_max": sum(entry["maximum_score"] for entry in graded),
    }
