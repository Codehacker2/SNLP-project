# AI Exam Answer Grader — Group 2 (SNLP)

Short-answer scoring component for the University AI Platform. Given a student
answer and a reference answer, it predicts a grade band (0 = incorrect,
1 = partially correct, 2 = correct), scales it to the question's marks, and
returns JSON feedback.

Dataset: [EngSAF](https://huggingface.co/datasets/IsmaelMousa/engsaf) — 4733 / 338 / 506 train / validation / test answers.

## Project structure

```
backend/
├── data/                       cleaned EngSAF splits (Week 2)
├── notebooks/
│   ├── load_data.ipynb             Week 2 – load + clean dataset
│   ├── baseline_tf_idf.ipynb       Week 2 – TF-IDF baseline
│   ├── sentence_embeddings.ipynb   Week 3 – MiniLM model + V1 JSON pipeline
│   └── week4_improved_model.ipynb  Week 4 – improved model + comparison
├── src/
│   ├── preprocessing.py        Week 2 – clean_text() + dataset download
│   ├── features.py             Week 4 – 7 hybrid features
│   ├── compare_models.py       Week 4 – trains Baseline / V1 / V2, writes comparison
│   └── grade.py                Week 4 – grade one answer with all three models
├── results/                    created by compare_models.py
└── models/                     created by compare_models.py
```

## Models

| Version | Week | Features | Classifier |
|---|---|---|---|
| Baseline | 2 | TF-IDF cosine similarity | Logistic Regression |
| V1 | 3 | MiniLM (all-MiniLM-L6-v2) cosine similarity | Logistic Regression |
| V2 (improved) | 4 | MiniLM cosine, TF-IDF cosine, key-point coverage, word overlap, 3 length features | Standardised, tuned Logistic Regression |

**Key-point coverage:** the reference answer is split into key points
(lines/sentences). For each key point we find the most similar sentence in the
student answer (MiniLM cosine) and average these best matches. Key points with
similarity below 0.5 are reported back as `missing_key_points` in the feedback.

## Setup and run

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
cd backend
uv sync                                   # install dependencies from uv.lock
uv run python src/compare_models.py       # train + compare (a few minutes on CPU)
uv run python src/grade.py                # demo: grade one answer with all 3 models
```

Custom answer:

```bash
uv run python src/grade.py --question "Q1" --max-marks 5 --reference "Supervised learning uses labelled data to learn a mapping from inputs to outputs." --answer "It learns from labelled examples."
```

Results are written to `results/model_comparison.md` (tables for the report)
and `results/model_comparison.csv`.

## Output format

```json
{
    "question": "Q4",
    "score": 10,
    "maximum_score": 10,
    "feedback": "The answer covers all core concepts present in the reference.",
    "confidence": 0.71,
    "model": "v2_hybrid",
    "missing_key_points": []
}
```

The first five fields are identical to the Week 3 V1 pipeline.
