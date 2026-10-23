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
let digitizedBooklet = null;

bookletForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  bookletSubmit.disabled = true;
  bookletStatus.textContent = "Reading booklet...";
  bookletResult.hidden = true;
  marksheet = null;
  digitizedBooklet = null;
  try {
    const files = Array.from(bookletForm.elements["booklet-files"].files);
    if (!files.length) throw new Error("Choose a PDF, page images, or a reviewed JSON file.");
    const answerKey = JSON.parse(await bookletForm.elements["answer-key"].files[0].text());
    if (files.length === 1 && /\.json$/i.test(files[0].name)) {
      digitizedBooklet = JSON.parse(await files[0].text());
    } else {
      if (files.some((file) => !/\.(pdf|png|jpe?g)$/i.test(file.name))) {
        throw new Error("Use one PDF or JPG/PNG page images; do not mix file types.");
      }
      if (files.some((file) => /\.pdf$/i.test(file.name)) && files.length !== 1) {
        throw new Error("Choose only one PDF, or select multiple JPG/PNG pages.");
      }
      if (files.reduce((total, file) => total + file.size, 0) > 60_000_000) {
        throw new Error("Page files must total 60 MB or less.");
      }
      if (files.length > 1) {
        const byPageName = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });
        files.sort((first, second) => byPageName.compare(first.name, second.name));
      }
      bookletStatus.textContent = "Digitizing scanned pages. This may take a few minutes...";
      const pages = await Promise.all(files.map(fileToBase64));
      const rosterText = bookletForm.elements.roster.value.trim();
      digitizedBooklet = await postBookletJson("/digitize-pages", {
        pages,
        booklet_id: bookletForm.elements["booklet-id"].value.trim() || files[0].name.replace(/\.[^.]+$/, ""),
        roster: rosterText ? rosterText.split(/[\s,]+/).filter(Boolean) : null,
      });
    }
    const transcriptPanel = document.querySelector("#transcript-panel");
    transcriptPanel.hidden = false;
    renderBookletList("#booklet-transcript", digitizedBooklet.answers || [],
      (entry) => `${entry.question}${entry.needs_review ? " (needs review)" : ""}: ${entry.answer_text}`);
    document.querySelector("#booklet-transcript-download").hidden = false;
    document.querySelector("#booklet-download").hidden = true;
    bookletResult.hidden = false;
    bookletStatus.textContent = "Grading eligible answers...";

    marksheet = await postBookletJson("/grade-booklet", { booklet: digitizedBooklet, answer_key: answerKey });
    document.querySelector("#booklet-summary").textContent =
      `${marksheet.booklet_id} · Student ${marksheet.student_id || "unknown"} · ${marksheet.auto_marks}/${marksheet.auto_max} marks graded · ${marksheet.review_queue.length} answers for review`;
    renderBookletList("#booklet-graded", marksheet.graded,
      (entry) => `${entry.question}: ${entry.score}/${entry.maximum_score} · ${entry.feedback || ""}`);
    renderBookletList("#booklet-review", marksheet.review_queue,
      (entry) => `${entry.question}: ${entry.reason}`);
    document.querySelector("#booklet-download").hidden = false;
    bookletStatus.textContent = "Booklet processed.";
  } catch (error) {
    bookletStatus.textContent = error instanceof TypeError
      ? "Could not connect to the grader. Start the backend at 127.0.0.1:8000."
      : error.message;
    if (digitizedBooklet) {
      document.querySelector("#booklet-summary").textContent =
        "Digitized booklet saved. Grading could not finish; download the transcript and retry.";
      renderBookletList("#booklet-graded", [], () => "");
      renderBookletList("#booklet-review", [], () => "");
    }
  } finally {
    bookletSubmit.disabled = false;
  }
});

async function postBookletJson(path, payload) {
  const response = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(typeof data.detail === "string" ? data.detail : "The booklet could not be processed.");
  }
  return data;
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1]);
    reader.onerror = () => reject(new Error(`Could not read ${file.name}.`));
    reader.readAsDataURL(file);
  });
}

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
  downloadJson(marksheet, `${marksheet?.booklet_id || "booklet"}_marksheet.json`);
});

document.querySelector("#booklet-transcript-download").addEventListener("click", () => {
  downloadJson(digitizedBooklet, `${digitizedBooklet?.booklet_id || "booklet"}_digitized.json`);
});

function downloadJson(data, name) {
  if (!data) return;
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}
