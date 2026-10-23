from contextlib import asynccontextmanager
import json
import hashlib
import os
from pathlib import Path
from typing import Literal, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from features import TextEncoder
from grade import V2_NLP_Pipeline
from booklet import GradeBookletRequest, grade_booklet

BACKEND_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BACKEND_DIR / "models" / "grader_models.joblib"

_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"{MODEL_PATH} not found. Run src/compare_models.py and src/improve_model.py first.")
    bundle = joblib.load(MODEL_PATH)
    _state["bundle"] = bundle
    _state["pipeline"] = V2_NLP_Pipeline(bundle, TextEncoder(bundle["embedding_model"]))
    yield
    _state.clear()


app = FastAPI(
    title="AI Exam Answer Grader",
    description=(
        "Group 2's reusable NLP asset for the AI University Platform."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8080",
        "http://localhost:8080",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class EvaluateAnswerRequest(BaseModel):
    """Evaluates Answer of the student against the reference answer."""

    question: str = Field(..., description="The exam question, or a question ID (e.g. 'Q4').", examples=["Q4 What does supervised learning use to train model?"])
    reference_answer: str = Field(..., description="The model/reference answer written by the instructor.")
    marking_rubric: Optional[str] = Field(
        default=None,
        description=(
            "Marking rubric / mark scheme, if the institution has one separate "
            "from the reference answer. Accepted for interface completeness."
        ),
    )
    student_answer: str = Field(..., description="The student's answer to grade.")
    maximum_marks: int = Field(..., ge=1, description="Marks available for this question, supplied by the frontend.")


class EvaluateAnswerResponse(BaseModel):
    """The response to an answer evaluation request."""

    question: str
    score: Optional[int] = None
    maximum_score: int
    feedback: Optional[str] = None
    confidence: Optional[float] = None
    missing_key_points: list[str] = Field(default_factory=list)
    model: str
    grading_status: Literal["graded", "human_review"] = "graded"
    promptguard: dict = Field(default_factory=dict)


ModelChoice = Literal["v3", "v2", "v1", "baseline"]


def _promptguard_config() -> tuple[str, float]:
    base_url = os.environ.get("PROMPTGUARD_URL", "http://127.0.0.1:8001").rstrip("/")
    try:
        timeout = float(os.environ.get("PROMPTGUARD_TIMEOUT_SECONDS", "5"))
    except ValueError as exc:
        raise HTTPException(500, "PROMPTGUARD_TIMEOUT_SECONDS must be a number.") from exc
    if timeout <= 0:
        raise HTTPException(500, "PROMPTGUARD_TIMEOUT_SECONDS must be greater than zero.")
    return base_url, timeout


def _screen_with_promptguard(text: str, request_id: str) -> dict:
    base_url, timeout = _promptguard_config()
    request = Request(
        f"{base_url}/v1/analyze",
        data=json.dumps({
            "text": text,
            "source_type": "exam_answer",
            "request_id": request_id,
            "include_sanitized": True,
            "include_signals": False,
        }).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except TimeoutError as exc:
        raise HTTPException(504, "PromptGuard did not respond before the configured timeout.") from exc
    except HTTPError as exc:
        detail = ""
        try:
            error_body = json.loads(exc.read().decode("utf-8"))
            if isinstance(error_body, dict) and error_body.get("detail"):
                detail = f": {error_body['detail']}"
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        raise HTTPException(
            502,
            f"PromptGuard rejected the screening request (HTTP {exc.code}){detail}",
        ) from exc
    except (URLError, OSError) as exc:
        raise HTTPException(
            502,
            "PromptGuard is unreachable. Start its local API and verify PROMPTGUARD_URL.",
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(502, "PromptGuard returned an invalid JSON response.") from exc

    if not isinstance(payload, dict) or "recommended_action" not in payload:
        raise HTTPException(502, "PromptGuard returned an unexpected response.")
    return payload


@app.get("/health")
def health():
    bundle = _state.get("bundle")
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model bundle not loaded.")
    return {
        "status": "ok",
        "embedding_model": bundle["embedding_model"],
        "available_models": [m for m in ["v3_model", "v2_model", "v1_clf", "baseline_clf"] if m in bundle],
    }


@app.post("/evaluate-answer", response_model=EvaluateAnswerResponse)
def evaluate_answer(req: EvaluateAnswerRequest, model: ModelChoice = "v3"):
    """
    Grade one student answer against a reference answer.

    `model` selects which trained approach to use (default `v3`, the best
    validated model):
      - `v3`: hybrid features (incl. keypoint_min_coverage, negation_mismatch) + tuned Logistic Regression
      - `v2`: Week 4 hybrid features + tuned Logistic Regression
      - `v1`: MiniLM sentence-embedding cosine similarity + Logistic Regression
      - `baseline`: TF-IDF cosine similarity + Logistic Regression
    """
    request_id = f"repo1-{hashlib.sha256(req.question.encode('utf-8')).hexdigest()[:16]}"
    promptguard_result = _screen_with_promptguard(req.student_answer, request_id)
    action = promptguard_result["recommended_action"]
    promptguard_summary = {
        "attack_detected": promptguard_result.get("attack_detected", False),
        "risk_level": promptguard_result.get("risk_level"),
        "recommended_action": action,
        "attack_type": promptguard_result.get("attack_type"),
        "sanitized_text": promptguard_result.get("sanitized_text"),
        "request_id": promptguard_result.get("request_id"),
    }
    if action in {"BLOCK", "HUMAN_REVIEW"}:
        return EvaluateAnswerResponse(
            question=req.question,
            score=None,
            maximum_score=req.maximum_marks,
            feedback="PromptGuard flagged this answer for human review; it was not automatically graded.",
            confidence=None,
            model="PromptGuard",
            grading_status="human_review",
            promptguard=promptguard_summary,
        )

    student_answer = req.student_answer
    if action == "SANITIZE":
        sanitized = promptguard_result.get("sanitized_text")
        if not isinstance(sanitized, str) or not sanitized.strip():
            raise HTTPException(502, "PromptGuard requested sanitization but returned no safe answer text.")
        student_answer = sanitized

    pipeline: V2_NLP_Pipeline = _state["pipeline"]
    dict_fn = {
        "v3": pipeline.grade_with_v3_dict,
        "v2": pipeline.grade_answer_dict,
        "v1": pipeline.grade_with_v1_dict,
        "baseline": pipeline.grade_with_baseline_dict,
    }[model]

    result = dict_fn(req.question, student_answer, req.reference_answer, req.maximum_marks)
    result.setdefault("missing_key_points", [])
    result["grading_status"] = "graded"
    result["promptguard"] = promptguard_summary
    return result


@app.post("/grade-booklet")
def evaluate_booklet(req: GradeBookletRequest, model: ModelChoice = "v3"):
    """Grade a reviewed Group 1 digitizer JSON using this service's grader."""
    def evaluate(question: str, reference: str, rubric: str, answer: str, marks: int) -> dict:
        result = evaluate_answer(
            EvaluateAnswerRequest(
                question=question,
                reference_answer=reference,
                marking_rubric=rubric,
                student_answer=answer,
                maximum_marks=marks,
            ),
            model=model,
        )
        return result.model_dump() if isinstance(result, BaseModel) else result

    try:
        return grade_booklet(req, evaluate)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
