const API_URL = "http://127.0.0.1:8000";

const form = document.querySelector("#grader-form");
const submitButton = document.querySelector("#submit-button");
const formStatus = document.querySelector("#form-status");
const result = document.querySelector("#result");
const scoreValue = document.querySelector("#score-value");
const scoreMaximum = document.querySelector("#score-maximum");
const confidenceValue = document.querySelector("#confidence-value");
const modelValue = document.querySelector("#model-value");
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
  const payload = {
    question: formData.get("question").trim(),
    reference_answer: formData.get("reference-answer").trim(),
    student_answer: formData.get("student-answer").trim(),
    maximum_marks: Number(formData.get("maximum-marks")),
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
  scoreValue.textContent = data.score;
  scoreMaximum.textContent = ` / ${data.maximum_score}`;
  confidenceValue.textContent = `${Math.round(data.confidence * 100)}%`;
  modelValue.textContent = data.model;
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
