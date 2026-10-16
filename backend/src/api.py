from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from features import TextEncoder
from grade import V2_NLP_Pipeline

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
    maximum_marks: int = Field(default=10, ge=1, description="Marks available for this question.")


class EvaluateAnswerResponse(BaseModel):
    """The response to an answer evaluation request."""

    question: str
    score: int
    maximum_score: int
    feedback: str
    confidence: float
    missing_key_points: list[str] = Field(default_factory=list)
    model: str


ModelChoice = Literal["v3", "v2", "v1", "baseline"]


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
    pipeline: V2_NLP_Pipeline = _state["pipeline"]
    dict_fn = {
        "v3": pipeline.grade_with_v3_dict,
        "v2": pipeline.grade_answer_dict,
        "v1": pipeline.grade_with_v1_dict,
        "baseline": pipeline.grade_with_baseline_dict,
    }[model]

    result = dict_fn(req.question, req.student_answer, req.reference_answer, req.maximum_marks)
    result.setdefault("missing_key_points", [])
    return result
