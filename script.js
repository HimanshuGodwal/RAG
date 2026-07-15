const askBtn = document.getElementById("askBtn");
const questionInput = document.getElementById("question");
const answerP = document.getElementById("answer");

const emptyState = document.getElementById("emptyState");
const transcript = document.getElementById("transcript");
const userQuestionText = document.getElementById("userQuestionText");

const API_URL = "http://127.0.0.1:8000/ask";

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
      body: JSON.stringify({ question: question }),
    });

    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}`);
    }

    const data = await response.json();
    answerP.textContent = data.answer;

  } catch (error) {
    answerP.textContent = `Error: ${error.message}. Is the FastAPI server running?`;
  } finally {
    askBtn.disabled = false;
    questionInput.focus();
  }
}

askBtn.addEventListener("click", submitQuestion);

questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") submitQuestion();
});
