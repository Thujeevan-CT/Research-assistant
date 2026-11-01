const questionEl = document.getElementById("question");
const submitEl = document.getElementById("submit");
const statusEl = document.getElementById("status");
const progressPanel = document.getElementById("progress");
const progressBody = document.getElementById("progress-body");
const answerPanel = document.getElementById("answer-panel");
const answerEl = document.getElementById("answer");
const transcriptLinkEl = document.getElementById("transcript-link");
const errorEl = document.getElementById("error");

let currentRoundEl = null;
let currentResultsEl = null;
let abortController = null;

submitEl.addEventListener("click", startResearch);
questionEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    startResearch();
  }
});

function resetUI() {
  progressBody.innerHTML = "";
  answerEl.innerHTML = "";
  answerEl.removeAttribute("data-filled");
  transcriptLinkEl.innerHTML = "";
  transcriptLinkEl.classList.add("hidden");
  errorEl.textContent = "";
  errorEl.classList.add("hidden");
  progressPanel.classList.add("hidden");
  answerPanel.classList.add("hidden");
  currentRoundEl = null;
  currentResultsEl = null;
}

function setBusy(busy) {
  submitEl.disabled = busy;
  questionEl.disabled = busy;
  statusEl.textContent = busy ? "Researching…" : "";
}

async function startResearch() {
  const question = questionEl.value.trim();
  if (!question) {
    showError("Please enter a question.");
    return;
  }

  if (abortController) {
    abortController.abort();
  }
  abortController = new AbortController();

  resetUI();
  setBusy(true);
  progressPanel.classList.remove("hidden");

  try {
    const response = await fetch("/api/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      signal: abortController.signal,
    });

    if (!response.ok) {
      throw new Error(await formatHttpError(response));
    }

    await consumeStream(response.body);
  } catch (err) {
    if (err.name === "AbortError") {
      return;
    }
    showError(err.message || "Something went wrong.");
  } finally {
    setBusy(false);
    abortController = null;
  }
}

async function consumeStream(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.trim()) {
        continue;
      }
      handleEvent(JSON.parse(line));
    }
  }

  if (buffer.trim()) {
    handleEvent(JSON.parse(buffer));
  }
}

function handleEvent(event) {
  switch (event.event) {
    case "started":
      statusEl.textContent = `Up to ${event.max_rounds} search rounds`;
      break;
    case "round":
      beginRound(event.number);
      break;
    case "search":
      addSearchLine(event.query);
      break;
    case "result":
      addResultLine(event.title);
      break;
    case "fetch":
      addFetchLine(event.url);
      break;
    case "fetched":
      addFetchResult(event.title);
      break;
    case "fetch_error":
      addFetchError(event.message);
      break;
    case "status":
      addStatusBanner(event.message);
      break;
    case "claim_check":
      addClaimCheck(event);
      break;
    case "grounding":
      statusEl.textContent = event.message || "Verifying claims…";
      break;
    case "answer":
      answerPanel.classList.remove("hidden");
      answerEl.innerHTML = formatAnswer(event.text);
      answerEl.dataset.filled = "1";
      statusEl.textContent = "Done";
      break;
    case "transcript":
      showTranscriptLink(event.run_id);
      break;
    case "error":
      showError(event.message || "Error");
      break;
    case "done":
      if (!answerEl.dataset.filled) {
        statusEl.textContent = "";
      }
      break;
    default:
      break;
  }
}

function beginRound(number) {
  const block = document.createElement("div");
  block.className = "round-block";
  block.innerHTML = `<p class="round-label">Round ${number}</p>`;
  currentRoundEl = block;
  currentResultsEl = null;
  progressBody.appendChild(block);
}

function addSearchLine(query) {
  if (!currentRoundEl) {
    beginRound("…");
  }
  const line = document.createElement("p");
  line.className = "search-line";
  line.innerHTML = `<strong>Search:</strong> ${escapeHtml(query)}`;
  currentRoundEl.appendChild(line);
  currentResultsEl = document.createElement("ul");
  currentResultsEl.className = "results";
  currentRoundEl.appendChild(currentResultsEl);
}

function addResultLine(title) {
  if (!currentResultsEl) {
    addSearchLine("…");
  }
  const item = document.createElement("li");
  item.textContent = title;
  currentResultsEl.appendChild(item);
}

function addFetchLine(url) {
  if (!currentRoundEl) {
    beginRound("…");
  }
  currentResultsEl = null;
  const line = document.createElement("p");
  line.className = "fetch-line";
  line.innerHTML = `<strong>Fetch:</strong> ${escapeHtml(url)}`;
  currentRoundEl.appendChild(line);
}

function addFetchResult(title) {
  if (!currentRoundEl) {
    beginRound("…");
  }
  const item = document.createElement("p");
  item.className = "fetch-result";
  item.textContent = `→ ${title}`;
  currentRoundEl.appendChild(item);
}

function addFetchError(message) {
  if (!currentRoundEl) {
    beginRound("…");
  }
  const item = document.createElement("p");
  item.className = "fetch-error";
  item.textContent = `→ ${message}`;
  currentRoundEl.appendChild(item);
}

function addStatusBanner(message) {
  const banner = document.createElement("div");
  banner.className = "status-banner";
  banner.textContent = message;
  progressBody.appendChild(banner);
}

function addClaimCheck(event) {
  if (!currentRoundEl) {
    beginRound("verify");
  }
  const line = document.createElement("p");
  line.className = event.pass ? "claim-pass" : "claim-fail";
  const mark = event.pass ? "✓" : "✗";
  line.textContent = `${mark} ${(event.claim || "").slice(0, 90)} — ${event.reason || ""}`;
  progressBody.appendChild(line);
}

function showTranscriptLink(runId) {
  transcriptLinkEl.innerHTML = `<a href="/api/transcripts/${escapeAttr(
    runId
  )}" download>Download transcript</a>`;
  transcriptLinkEl.classList.remove("hidden");
}

function showError(message) {
  errorEl.textContent = message;
  errorEl.classList.remove("hidden");
  statusEl.textContent = "";
}

async function formatHttpError(response) {
  const text = await response.text();
  try {
    const data = JSON.parse(text);
    if (Array.isArray(data.detail)) {
      const msg = data.detail
        .map((item) => item.msg || JSON.stringify(item))
        .join("; ");
      return msg || `Request failed (${response.status})`;
    }
    if (typeof data.detail === "string") {
      return data.detail;
    }
  } catch {
    /* plain text */
  }
  return text || `Request failed (${response.status})`;
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function escapeAttr(text) {
  return escapeHtml(text).replace(/'/g, "&#39;");
}

function renderInline(text) {
  return escapeHtml(text).replace(
    /\*\*(.+?)\*\*/g,
    "<strong>$1</strong>"
  );
}

function markdownBlock(text) {
  const lines = text.split("\n");
  let html = "";
  let inList = false;

  const closeList = () => {
    if (inList) {
      html += "</ul>";
      inList = false;
    }
  };

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      closeList();
      continue;
    }

    const heading = trimmed.match(/^#{1,3}\s+(.*)$/);
    if (heading) {
      closeList();
      html += `<h3 class="answer-heading">${renderInline(heading[1])}</h3>`;
      continue;
    }

    const bullet = trimmed.match(/^[*-]\s+(.*)$/);
    if (bullet) {
      if (!inList) {
        html += '<ul class="answer-list">';
        inList = true;
      }
      html += `<li>${renderInline(bullet[1])}</li>`;
      continue;
    }

    closeList();
    html += `<p>${renderInline(trimmed)}</p>`;
  }

  closeList();
  return html;
}

function parseSourceLine(line) {
  const cleaned = line.replace(/^[*-]\s*/, "").trim();
  if (!cleaned) {
    return "";
  }

  const linked = cleaned.match(/^(.+?)\s*[—–]\s*(https?:\/\/\S+)\s*$/);
  if (linked) {
    const title = linked[1].trim();
    const url = linked[2].trim();
    return `<li><a href="${escapeAttr(url)}" target="_blank" rel="noopener noreferrer">${renderInline(title)}</a></li>`;
  }

  const urlOnly = cleaned.match(/(https?:\/\/\S+)/);
  if (urlOnly) {
    const url = urlOnly[1];
    const title = cleaned.replace(url, "").replace(/[—–-]\s*$/, "").trim() || url;
    return `<li><a href="${escapeAttr(url)}" target="_blank" rel="noopener noreferrer">${renderInline(title)}</a></li>`;
  }

  return `<li>${renderInline(cleaned)}</li>`;
}

function formatAnswer(raw) {
  let text = raw.trim();
  text = text.replace(/^1\.\s*A short summary:?\s*/i, "");
  text = text.replace(/^\*\*Summary\*\*:?\s*/im, "");

  const split = text.split(/\n\s*(?:#{1,3}\s*)?Sources:?\s*\n/i);
  let summaryPart = split[0] || "";
  let sourcesPart = split[1] || "";

  summaryPart = summaryPart
    .replace(/^#{1,3}\s*Summary\s*\n?/im, "")
    .trim();

  if (!sourcesPart && /\n[-*]\s+.+[—–].+https?:\/\//.test(text)) {
    const idx = text.search(/\n\s*[-*]\s+.+[—–]/);
    if (idx > 0) {
      summaryPart = text.slice(0, idx).trim();
      sourcesPart = text.slice(idx).trim();
    }
  }

  const summaryHtml = markdownBlock(summaryPart);
  const sourceLines = sourcesPart
    .split("\n")
    .map((line) => parseSourceLine(line))
    .filter(Boolean)
    .join("");

  let html = `<div class="answer-summary">${summaryHtml}</div>`;
  if (sourceLines) {
    html += `<h3 class="answer-heading answer-sources-heading">Sources</h3>`;
    html += `<ul class="source-list">${sourceLines}</ul>`;
  }

  return html;
}
