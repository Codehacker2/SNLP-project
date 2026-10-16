# AI Exam Answer Grader

The backend scores a short student answer against an instructor/reference
answer. It predicts one of three grade bands and scales that prediction to the
question's maximum marks:

| Label | Meaning |
| --- | --- |
| `0` | Incorrect |
| `1` | Partially correct |
| `2` | Correct |

The project uses the [EngSAF dataset](https://huggingface.co/datasets/IsmaelMousa/engsaf)
and exposes the trained grader through a FastAPI service. The current default is
the V3 hybrid model, which adds negation mismatch detection and minimum
key-point coverage to the earlier hybrid feature set.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- Internet access on the first run, so `sentence-transformers` can download
  `all-MiniLM-L6-v2` and the preprocessing script can download EngSAF

Dependencies and exact resolved versions are declared in
[`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock).

## Installation

From the repository root:

```bash
cd backend
uv sync
```

The repository includes a trained model bundle at
[`models/grader_models.joblib`](models/grader_models.joblib), so the API and
grading CLI can be used immediately after installing dependencies.

## Run the API

Start the development server from the `backend` directory:

```bash
uv run uvicorn api:app --app-dir src --reload
```

The `--app-dir src` option is required because the source modules currently use
local imports such as `from features import TextEncoder`.

The service is available at `http://127.0.0.1:8000`. FastAPI's interactive
documentation is available at:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### Health check

```bash
curl http://127.0.0.1:8000/health
```

Example response:

```json
{
  "status": "ok",
  "embedding_model": "all-MiniLM-L6-v2",
  "available_models": ["v3_model", "v2_model", "v1_clf", "baseline_clf"]
}
```

### Evaluate an answer

`POST /evaluate-answer` accepts the following JSON body:

```json
{
  "question": "Q4",
  "reference_answer": "Supervised learning uses labelled data to learn a mapping from inputs to outputs.",
  "marking_rubric": "Award full marks for identifying labelled training data and the learned mapping.",
  "student_answer": "It learns from labelled examples.",
  "maximum_marks": 5
}
```

The `marking_rubric` field is accepted for interface compatibility but is not
currently used by the model. To select a model, add the `model` query
parameter: `v3` (default), `v2`, `v1`, or `baseline`.

```bash
curl -X POST "http://127.0.0.1:8000/evaluate-answer?model=v3" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"Q4\",\"reference_answer\":\"Supervised learning uses labelled data to learn a mapping from inputs to outputs.\",\"student_answer\":\"It learns from labelled examples.\",\"maximum_marks\":5}"
```

Response shape:

```json
{
  "question": "Q4",
  "score": 3,
  "maximum_score": 5,
  "feedback": "The answer is partially correct but misses some key concepts.",
  "confidence": 0.71,
  "missing_key_points": [],
  "model": "v3_improved"
}
```

The model predicts a grade band first, so scores are rounded from
`(predicted_label / 2) * maximum_marks`. `missing_key_points` contains up to
three reference points whose semantic similarity is below the explainability
threshold.

## Run the command-line grader

From `backend`:

```bash
uv run python src/grade.py
```

This prints results from all four available model versions for a built-in
example. To grade a custom answer:

```bash
uv run python src/grade.py `
  --question "Q1" `
  --max-marks 5 `
  --reference "Supervised learning uses labelled data to learn a mapping from inputs to outputs." `
  --answer "It learns from labelled examples."
```

The backtick line continuation is for PowerShell. On macOS/Linux, use `\`
instead.

## Train or regenerate artifacts

### Prepare the dataset

The cleaned EngSAF CSV files are already stored in [`data/`](data/). To
download and regenerate them, run the preprocessing script from `backend/src`
so its `../data` output resolves to the backend data directory:

```bash
cd backend/src
uv run python preprocessing.py
cd ..
```

This creates:

- `data/engsaf_train.csv`
- `data/engsaf_validation.csv`
- `data/engsaf_test.csv`

### Train the baseline, V1, and V2 models

```bash
cd backend
uv run python src/compare_models.py
```

This trains and evaluates:

1. **Baseline**: TF-IDF cosine similarity plus logistic regression.
2. **V1**: `all-MiniLM-L6-v2` sentence-embedding cosine similarity plus
   logistic regression.
3. **V2**: a tuned logistic regression over MiniLM similarity, TF-IDF
   similarity, key-point coverage, word overlap, and answer-length features.

The command writes the model bundle to
[`models/grader_models.joblib`](models/grader_models.joblib) and reports to
[`results/model_comparison.md`](results/model_comparison.md) and
[`results/model_comparison.csv`](results/model_comparison.csv).

### Train the V3 improvement

```bash
uv run python src/improve_model.py
```

V3 replaces average key-point coverage with minimum coverage and adds a
`negation_mismatch` feature. It searches logistic regression, random forest,
and gradient boosting candidates using validation macro-F1, then appends the
selected V3 model to the existing model bundle. Its report is written to
[`results/model_comparison_v3.md`](results/model_comparison_v3.md).

The test set is used only after selecting the candidate on the validation set.

## Current benchmark

The V3 results below are from the stored EngSAF test split (506 examples):

| Model | Accuracy | Macro-F1 | QWK | MAE | Partial recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| V2 hybrid | 0.5257 | 0.5005 | 0.3643 | 0.5711 | 0.3770 |
| V3 improved | 0.5494 | 0.5252 | 0.3919 | 0.5415 | 0.4084 |

See [`results/model_comparison_v3.md`](results/model_comparison_v3.md) for
the classifier search and validation metrics. See
[`results/model_comparison.md`](results/model_comparison.md) for the complete
baseline/V1/V2 comparison, confusion matrices, ablation study, and feature
weights.

## Project structure

```text
backend/
├── data/                         Cleaned EngSAF train/validation/test CSVs
├── models/
│   └── grader_models.joblib      Trained baseline, V1, V2, and V3 bundle
├── notebooks/                    Dataset and model development notebooks
├── results/                      Evaluation tables and predictions
├── src/
│   ├── api.py                    FastAPI application
│   ├── compare_models.py         Baseline/V1/V2 training and evaluation
│   ├── evaluate_v3.py            V3 evaluation
│   ├── features.py               TF-IDF, MiniLM, hybrid, and V3 features
│   ├── grade.py                  Inference pipeline and CLI
│   ├── improve_model.py          V3 model search and training
│   └── preprocessing.py          EngSAF download and cleaning
├── main.py                       Minimal package entry point
├── pyproject.toml                Project metadata and dependencies
└── uv.lock                      Locked dependency resolution
```

## Notes

- The API loads the model bundle during application startup. If the bundle is
  missing, run `compare_models.py` and then `improve_model.py`.
- Model inference uses CPU by default and the first MiniLM load can take
  longer while the transformer model is downloaded and cached.
- This is a three-class short-answer grader, not a generative feedback model.
  Feedback is generated from the predicted class and unmatched reference
  key points.
