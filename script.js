import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";
import { MENTAL_HEALTH_KB } from "./mental_health_kb.js";

/* ---------------- ENV ---------------- */
env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;

/* ---------------- DOM ---------------- */
const chatContainer = document.getElementById("chatContainer");
const userInput = document.getElementById("userInput");
const sendButton = document.getElementById("sendButton");
const statusMessage = document.getElementById("statusMessage");

/* ---------------- STATE ---------------- */
let embedder;
let kbEmbeddings = [];
let ready = false;

/* ---------------- CRISIS ---------------- */
const crisisKeywords = [
  "suicide", "kill myself", "end it all",
  "hurt myself", "self harm", "want to die"
];

function containsCrisis(text) {
  return crisisKeywords.some(k => text.toLowerCase().includes(k));
}

function crisisResponse() {
  return `🚨 Your safety matters.

Please reach out immediately:
• Crisis Text Line: Text HOME to 741741
• Suicide Prevention Lifeline: 988

You are not alone. 💜`;
}

/* ---------------- INTENT ---------------- */
function detectIntent(text) {
  const t = text.toLowerCase().trim();

  if (t.length <= 3 || ["hi","hello","hey","yo","hii"].includes(t)) {
    return "greeting";
  }

  if (["wtf","ayein","abeee","bhai","???"].some(k => t.includes(k))) {
    return "confusion";
  }

  if (["fuck","shit","bc","mc","nigga"].some(k => t.includes(k))) {
    return "abuse";
  }

  if (t.split(" ").length <= 2) {
    return "low_info";
  }

  return "emotional";
}

const INTENT_RESPONSES = {
  greeting: [
    "Hey 👋 I’m here with you. What’s been on your mind lately?",
    "Hi! Tell me what’s going on."
  ],
  confusion: [
    "😅 Looks like something didn’t land right. Want to explain?",
    "I’m listening — what just happened?"
  ],
  abuse: [
    "I’m here to help, but let’s keep it respectful. What’s really bothering you?",
    "Sounds like a lot of frustration — want to talk about it?"
  ],
  low_info: [
    "I’m listening 🙂 Can you share a bit more?",
    "Take your time — what’s going on for you?"
  ]
};

/* ---------------- CATEGORY ---------------- */
const categories = {
  stress_academic: ["exam","deadline","study","grades","academic"]
};

function detectCategory(text) {
  const t = text.toLowerCase();
  for (const [c, keys] of Object.entries(categories)) {
    if (keys.some(k => t.includes(k))) return c;
  }
  return "general_support";
}

/* ---------------- RAG ---------------- */
function cosine(a, b) {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return d / (Math.sqrt(na) * Math.sqrt(nb));
}

async function retrieveContext(text, category) {
  const q = await embedder(text, { pooling: "mean", normalize: true });

  return kbEmbeddings
    .filter(k => k.category === category)
    .map(k => ({ ...k, score: cosine(q.data, k.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 3);
}

/* ---------------- LANGUAGE ---------------- */
function detectLanguage(text) {
  const hindiHints = ["bhai","yaar","samajh","thoda","kya","ayein","abey"];
  return hindiHints.some(w => text.toLowerCase().includes(w))
    ? "hinglish"
    : "english";
}

/* ---------------- PROMPT ---------------- */
function buildSystemPrompt(language) {
  let prompt = `
You are a warm, emotionally intelligent mental wellness companion.

Rules:
- Sound natural and conversational (like ChatGPT)
- Validate emotions first
- Ask gentle follow-up questions
- 2–4 sentences max
- No diagnosis or medical advice
- Do not repeat phrasing
`;

  if (language === "hinglish") {
    prompt += `
Use Hinglish (Hindi + English mix).
Casual, friendly Indian tone.
`;
  }

  return prompt;
}

function buildContext(rag) {
  if (!rag.length) return "";
  return `
Background emotional context (use subtly):
${rag.map(r => "- " + r.text).join("\n")}
`;
}

/* ---------------- SERVER CALL ---------------- */
async function callLLM(messages) {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages })
  });

  const data = await res.json();

  if (!data || !data.content) {
    return "Samajh raha hoon. Thoda aur bataoge kya ho raha hai?";
  }

  return data.content.trim();
}

/* ---------------- INIT ---------------- */
async function initAI() {
  statusMessage.textContent = "Loading AI locally… 🧠";

  embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");

  for (const item of MENTAL_HEALTH_KB) {
    const e = await embedder(item.text, { pooling: "mean", normalize: true });
    kbEmbeddings.push({ ...item, embedding: e.data });
  }

  ready = true;
  statusMessage.textContent = "AI ready • Secure cloud responses 🔒";
}
initAI();

/* ---------------- UI ---------------- */
function addMessage(sender, text) {
  const div = document.createElement("div");
  div.className = `message ${sender}`;
  div.textContent = text;
  chatContainer.appendChild(div);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

/* ---------------- MAIN ---------------- */
async function handleMessage() {
  if (!ready) return;

  const text = userInput.value.trim();
  if (!text) return;

  addMessage("user", text);
  userInput.value = "";

  let reply;

  if (containsCrisis(text)) {
    reply = crisisResponse();
  } else {
    const intent = detectIntent(text);

    if (intent !== "emotional") {
      const options = INTENT_RESPONSES[intent];
      reply = options[Math.floor(Math.random() * options.length)];
    } else {
      const category = detectCategory(text);
      const rag = await retrieveContext(text, category);
      const language = detectLanguage(text);

      const messages = [
        { role: "system", content: buildSystemPrompt(language) },
        ...(rag.length ? [{ role: "system", content: buildContext(rag) }] : []),
        { role: "user", content: text }
      ];

      reply = await callLLM(messages);
    }
  }

  addMessage("assistant", reply);
}

sendButton.onclick = handleMessage;
userInput.addEventListener("keydown", e => {
  if (e.key === "Enter") handleMessage();
});
