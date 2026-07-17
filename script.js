const askBtn = document.getElementById("askBtn");
const questionInput = document.getElementById("question");
const answerP = document.getElementById("answer");
const sourcesList = document.getElementById("sourcesList");

const emptyState = document.getElementById("emptyState");
const transcript = document.getElementById("transcript");
const userQuestionText = document.getElementById("userQuestionText");

const pdfInput = document.getElementById("pdfInput");
const uploadBtn = document.getElementById("uploadBtn");
const uploadStatus = document.getElementById("uploadStatus");

const API_URL = "http://127.0.0.1:8000/ask";
const UPLOAD_URL = "http://127.0.0.1:8000/upload";

// One random id per browser tab, so uploads made here don't
// show up when someone else uses the app at the same time.
let sessionId = sessionStorage.getItem("sessionId");
if (!sessionId) {
  sessionId = crypto.randomUUID();
  sessionStorage.setItem("sessionId", sessionId);
}

const themeToggle = document.getElementById("themeToggle");





themeToggle.addEventListener("click", () => {
  const isLight = document.documentElement.getAttribute("data-theme") === "light";
  document.documentElement.setAttribute("data-theme", isLight ? "dark" : "light");
});



async function submitQuestion() {
  const question = questionInput.value.trim();
  if (!question) return;

  
  emptyState.hidden = true;
  transcript.hidden = false;
  userQuestionText.textContent = question;

  questionInput.value = "";
  askBtn.disabled = true;
  answerP.textContent = "Thinking...";


  
  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question: question, session_id: sessionId }),
    });

    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}`);
    }

    const data = await response.json();
    answerP.textContent = data.answer;
    renderSources(data.sources);

  } catch (error) {
    answerP.textContent = `Error: ${error.message}. Is the FastAPI server running?`;
    renderSources([]);
  } finally {
    askBtn.disabled = false;
    questionInput.focus();
  }
}

function renderSources(sources) {
  if (!sources || sources.length === 0) {
    sourcesList.innerHTML = "";
    return;
  }

  sourcesList.innerHTML = sources
    .map(
      (s) =>
        `<div class="source-item">${s.source} · page ${s.page} · ${s.match}% match</div>`
    )
    .join("");
}

async function uploadPdf() {
  const file = pdfInput.files[0];
  if (!file) {
    uploadStatus.textContent = "Choose a PDF first.";
    return;
  }

  uploadBtn.disabled = true;
  uploadStatus.textContent = "Indexing...";

  const formData = new FormData();
  formData.append("file", file);
  formData.append("session_id", sessionId);

  try {
    const response = await fetch(UPLOAD_URL, {
      method: "POST",
      body: formData, // no Content-Type header — the browser sets the
                       // multipart boundary itself when the body is FormData
    });

    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}`);
    }

    const data = await response.json();

    if (data.error) {
      uploadStatus.textContent = data.error;
    } else {
      uploadStatus.textContent = `Added ${data.filename} (${data.chunks_added} chunks)`;
    }

  } catch (error) {
    uploadStatus.textContent = `Error: ${error.message}`;
  } finally {
    uploadBtn.disabled = false;
  }
}

uploadBtn.addEventListener("click", uploadPdf);

askBtn.addEventListener("click", submitQuestion);

questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") submitQuestion();
});
