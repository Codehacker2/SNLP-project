const API_URL = "http://127.0.0.1:8000";

const form = document.querySelector("#grader-form");
const submitButton = document.querySelector("#submit-button");
const formStatus = document.querySelector("#form-status");
const result = document.querySelector("#result");
const resultTitle = document.querySelector("#result-title");
const scoreValue = document.querySelector("#score-value");
const scoreMaximum = document.querySelector("#score-maximum");
const confidenceValue = document.querySelector("#confidence-value");
const modelValue = document.querySelector("#model-value");
const promptguardValue = document.querySelector("#promptguard-value");
const feedbackValue = document.querySelector("#feedback-value");
const missingBlock = document.querySelector("#missing-block");
const missingPoints = document.querySelector("#missing-points");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setLoading(true);
  formStatus.textContent = "";
  result.hidden = true;

  const formData = new FormData(form);
  const model = formData.get("model");
  const maximumMarks = Number(formData.get("maximum-marks"));
  if (!Number.isInteger(maximumMarks) || maximumMarks < 1) {
    formStatus.textContent = "Enter a whole-number maximum score of at least 1.";
    setLoading(false);
    return;
  }
  const payload = {
    question: formData.get("question").trim(),
    reference_answer: formData.get("reference-answer").trim(),
    student_answer: formData.get("student-answer").trim(),
    maximum_marks: maximumMarks,
  };

  try {
    const response = await fetch(`${API_URL}/evaluate-answer?model=${model}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "The answer could not be graded.");
    }

    renderResult(data);
    formStatus.textContent = "Answer graded.";
  } catch (error) {
    formStatus.textContent =
      error instanceof TypeError
        ? "Could not connect to the grader. Start the backend at 127.0.0.1:8000."
        : error.message;
  } finally {
    setLoading(false);
  }
});

function renderResult(data) {
  resultTitle.textContent =
    data.grading_status === "human_review" ? "Human review required" : "Answer graded";
  scoreValue.textContent = data.score === null ? "—" : data.score;
  scoreMaximum.textContent = ` / ${data.maximum_score}`;
  confidenceValue.textContent =
    data.confidence === null ? "Not graded" : `${Math.round(data.confidence * 100)}%`;
  modelValue.textContent = data.model;
  const guard = data.promptguard || {};
  promptguardValue.textContent = guard.recommended_action
    ? `${guard.recommended_action} (${guard.risk_level || "unknown"})`
    : "Not available";
  feedbackValue.textContent = data.feedback;

  missingPoints.replaceChildren();
  for (const point of data.missing_key_points || []) {
    const item = document.createElement("li");
    item.textContent = point;
    missingPoints.append(item);
  }
  missingBlock.hidden = !data.missing_key_points?.length;
  result.hidden = false;
}

function setLoading(isLoading) {
  submitButton.disabled = isLoading;
  submitButton.textContent = isLoading ? "Grading..." : "Grade answer";
}

const bookletForm = document.querySelector("#booklet-form");
const bookletStatus = document.querySelector("#booklet-status");
const bookletResult = document.querySelector("#booklet-result");
const bookletSubmit = document.querySelector("#booklet-submit");
let marksheet = null;

bookletForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  bookletSubmit.disabled = true;
  bookletStatus.textContent = "Grading booklet...";
  bookletResult.hidden = true;
  try {
    const booklet = JSON.parse(await bookletForm.elements.booklet.files[0].text());
    const answerKey = JSON.parse(await bookletForm.elements["answer-key"].files[0].text());
    const response = await fetch(`${API_URL}/grade-booklet`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ booklet, answer_key: answerKey }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(typeof data.detail === "string" ? data.detail : "The booklet could not be graded.");
    }
    marksheet = data;
    document.querySelector("#booklet-summary").textContent =
      `${data.booklet_id} · Student ${data.student_id || "unknown"} · ${data.auto_marks}/${data.auto_max} marks graded · ${data.review_queue.length} answers for review`;
    renderBookletList("#booklet-graded", data.graded,
      (entry) => `${entry.question}: ${entry.score}/${entry.maximum_score} · ${entry.feedback || ""}`);
    renderBookletList("#booklet-review", data.review_queue,
      (entry) => `${entry.question}: ${entry.reason}`);
    bookletResult.hidden = false;
    bookletStatus.textContent = "Booklet processed.";
  } catch (error) {
    bookletStatus.textContent = error instanceof TypeError
      ? "Could not connect to the grader. Start the backend at 127.0.0.1:8000."
      : error.message;
  } finally {
    bookletSubmit.disabled = false;
  }
});

function renderBookletList(selector, entries, describe) {
  const list = document.querySelector(selector);
  list.replaceChildren();
  for (const entry of entries) {
    const item = document.createElement("li");
    item.textContent = describe(entry);
    list.append(item);
  }
  if (!entries.length) {
    const item = document.createElement("li");
    item.textContent = "None";
    list.append(item);
  }
}

document.querySelector("#booklet-download").addEventListener("click", () => {
  if (!marksheet) return;
  const blob = new Blob([JSON.stringify(marksheet, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${marksheet.booklet_id || "booklet"}_marksheet.json`;
  link.click();
  URL.revokeObjectURL(url);
});
