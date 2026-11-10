// If the frontend is hosted separately from the Flask API, set window.API_BASE
// (e.g. via ?api=http://127.0.0.1:5000) — otherwise it uses the same origin.
const API_BASE = (new URLSearchParams(location.search).get("api") || window.API_BASE || "").replace(/\/$/, "");

const stage = document.getElementById("stage");
const chat = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const statusEl = document.getElementById("status");
const statusText = document.getElementById("status-text");
const startersEl = document.getElementById("starters");

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
// Static markup (no user data), so setting innerHTML is safe here.
const ARROW = '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" focusable="false">' +
  '<path d="M5 12h14M13 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;   // textContent => no HTML injection
  return node;
}

function scrollDown() {
  stage.scrollTo({ top: stage.scrollHeight, behavior: reduceMotion.matches ? "auto" : "smooth" });
}

// The first message tucks the intro away so the conversation gets the whole screen.
function startChat() { document.body.classList.add("chatting"); }

function addUser(text) {
  startChat();
  chat.appendChild(el("div", "msg user", text));
  scrollDown();
}

function addBot(text, extras, kind) {      // kind: "error" adds a red edge
  startChat();
  const box = el("div", kind ? `msg bot ${kind}` : "msg bot");
  box.appendChild(el("p", null, text));
  if (extras) extras(box);
  chat.appendChild(box);
  scrollDown();
  return box;
}

// Three softly pulsing dots while the answer is being looked up.
function addTyping() {
  startChat();
  const box = el("div", "msg bot typing");
  box.setAttribute("role", "status");
  box.setAttribute("aria-label", "Searching FAQs");
  for (let i = 0; i < 3; i++) box.appendChild(el("i"));
  chat.appendChild(box);
  scrollDown();
  return box;
}

// Count the similarity number up from 0 while the bar fills.
function animateScore(node, target) {
  if (reduceMotion.matches) { node.textContent = target.toFixed(2); return; }
  const start = performance.now(), duration = 800;
  node.textContent = "0.00";
  const tick = (now) => {
    const t = Math.min(1, (now - start) / duration);
    node.textContent = (target * (1 - Math.pow(1 - t, 3))).toFixed(2);
    if (t < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function confidenceMeta(box, data) {
  const meta = el("div", "meta");
  if (data.match) {
    const closest = el("div", "closest");
    closest.append(el("span", "k", "Closest FAQ"), el("span", "matched-q", data.match.question));
    meta.appendChild(closest);

    const conf = el("div", `conf ${data.confidence}`);
    const bar = el("div", "bar");
    const fill = el("span");
    fill.style.setProperty("--v", Math.min(1, data.match.score * 1.6));   // stretch for visibility
    bar.appendChild(fill);
    // The score is word overlap with the FAQ, not a guarantee that the FAQ answers the question.
    const label = data.confidence === "high" ? "Strong match" : "Possible match";
    const sim = el("span", "sim", "word similarity ");
    const score = el("span", "score", data.match.score.toFixed(2));
    sim.appendChild(score);
    conf.append(bar, el("span", "badge", label), sim);
    meta.appendChild(conf);

    // Start the fill and the count-up on the next painted frame so the transition runs.
    requestAnimationFrame(() => requestAnimationFrame(() => {
      conf.classList.add("go");
      animateScore(score, data.match.score);
    }));
  }
  box.appendChild(meta);
}

// Shown when nothing matched closely enough: a red dot plus text, so it never relies on colour alone.
function noMatchMeta(box) {
  const meta = el("div", "meta nomatch");
  const conf = el("div", "conf low go");
  conf.appendChild(el("span", "badge", "No close match"));
  meta.appendChild(conf);
  box.appendChild(meta);
}

function suggestionChips(box, suggestions, title) {
  if (!suggestions.length) return;
  const wrap = el("div", "related");
  wrap.appendChild(el("p", "label", title));
  const chips = el("div", "chips");
  suggestions.forEach((s) => {
    const chip = el("button", "chip", s.question);
    chip.type = "button";
    chip.addEventListener("click", () => showFaq(s));
    chips.appendChild(chip);
  });
  wrap.appendChild(chips);
  box.appendChild(wrap);
}

async function api(path, options) {
  if (window.LOCAL_API) return window.LOCAL_API(path, options);   // standalone page: no server
  const res = await fetch(API_BASE + path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

async function showFaq(s) {
  addUser(s.question);
  try {
    const faq = await api(`/api/faqs/${s.id}`);
    addBot(faq.answer);
  } catch (err) {
    addBot(`Something went wrong: ${err.message}`, null, "error");
  }
}

async function ask(question) {
  addUser(question);
  const typing = addTyping();
  sendBtn.disabled = true;
  try {
    const data = await api("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: 4 }),
    });
    typing.remove();
    addBot(data.answer, (box) => {
      if (data.matched) confidenceMeta(box, data);
      else noMatchMeta(box);
      suggestionChips(
        box,
        data.suggestions,
        data.matched ? "Not what you meant? Try:" : "Did you mean…"
      );
    });
  } catch (err) {
    typing.remove();
    addBot(`Couldn't reach the FAQ service: ${err.message}`, null, "error");
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q) return;
  input.value = "";
  ask(q);
});

async function init() {
  try {
    const [health, faqs] = await Promise.all([api("/api/health"), api("/api/faqs")]);
    statusText.textContent = `${health.faq_count} FAQs loaded`;
    statusEl.className = "status ok";
    // A few example questions: one per category, up to 4, kept short and calm.
    const seen = new Set();
    faqs.faqs.forEach((f) => {
      if (seen.size >= 4 || seen.has(f.category)) return;
      seen.add(f.category);
      const row = el("button", "starter");
      row.type = "button";
      row.append(el("span", null, f.question));
      const arrow = el("span", "arr");
      arrow.innerHTML = ARROW;
      row.appendChild(arrow);
      row.addEventListener("click", () => ask(f.question));
      startersEl.appendChild(row);
    });
  } catch (err) {
    statusText.textContent = "backend offline";
    statusEl.className = "status err";
    addBot("I can't reach the backend right now. Start it with `python backend/app.py` and refresh.", null, "error");
  }
}
init();
