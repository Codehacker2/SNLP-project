# AI Exam Answer Grader + Handwritten Digitizer

This workspace combines Group 2's grader with Group 1's handwritten exam digitizer. The original grader lives in [`backend/`](backend/) and its browser page in [`frontend/`](frontend/). The digitizer source, review page, sample booklet and its own documentation live in [`digitizer/`](digitizer/).

## Run the combined flow

Use Python 3.11 or newer. Install and start each service in its own terminal:

```bash
cd digitizer
pip install -e ".[api,pdf]"
exam-digitizer serve --port 8002
```

```bash
cd backend
uv sync
uv run uvicorn api:app --app-dir src --port 8000
```

The grader also requires the PromptGuard service described in [`backend/README.md`](backend/README.md); set `PROMPTGUARD_URL` if it is not on `http://127.0.0.1:8001`. The first run may download the digitizer's TrOCR model and the grader's sentence encoder.

1. Open the digitizer review page at `http://127.0.0.1:8002/ui`, upload scanned pages, check the transcript and export reviewed JSON.
2. Run `python -m http.server 5500 --directory frontend` from the repository root and open `http://127.0.0.1:5500`.
3. In **Grade a digitized booklet**, select the reviewed JSON and an answer key such as [`digitizer/demo/answer_key.json`](digitizer/demo/answer_key.json). Download the resulting marksheet JSON.

The browser sends both files to `POST /grade-booklet?model=v3`. The request body is:

```json
{"booklet": {"schema_version": "1.0", "booklet_id": "B1", "student_id": "S1", "answers": [{"question": "Q1", "answer_text": "example", "page": 1, "confidence": 0.9, "needs_review": false, "flags": []}]}, "answer_key": {"Q1": {"question_text": "Question one", "reference_answer": "example", "rubric": ["example"], "max_marks": 5}}}
```

The adapter maps `max_marks` to the grader's `maximum_marks`, joins rubric points into `marking_rubric`, and calls the existing `/evaluate-answer` logic for each eligible answer. Answers with no student ID, an unresolved digitizer review flag, an unknown question, an empty transcript, a duplicate question, or a PromptGuard review decision go to `review_queue`. The response includes graded answers, combined confidence, automatic totals and missing questions. A PromptGuard outage returns an API error rather than silently grading unscreened answers.

For the individual services and their model documentation, see [`backend/README.md`](backend/README.md) and [`digitizer/README.md`](digitizer/README.md).
