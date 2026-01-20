import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";
import { MENTAL_HEALTH_KB } from "./mental_health_kb.js";

/* ---------- DOM ---------- */
const chatContainer = document.getElementById("chatContainer");
const userInput = document.getElementById("userInput");
const sendButton = document.getElementById("sendButton");
const statusMessage = document.getElementById("statusMessage");

/* ---------- AI STATE ---------- */
let embedder, generator;
let kbEmbeddings = [];

/* ---------- CRISIS ---------- */
const crisisKeywords = [
  "suicide","kill myself","end it all","hurt myself","self harm","want to die"
];

function containsCrisis(text) {
  return crisisKeywords.some(k => text.toLowerCase().includes(k));
}

function crisisResponse() {
  return `🚨 Your safety matters.

Please reach out immediately:
• Crisis Text Line: Text HOME to 741741
• Suicide Prevention Lifeline: 988

You don’t have to face this alone. 💜`;
}

/* ---------- CATEGORY ---------- */
const categories = {
  stress_academic: ["exam","deadline","study","grades"],
  anxiety_worry: ["anxious","panic","worried"],
  sadness_depression: ["sad","empty","hopeless"],
  self_esteem: ["worthless","stupid","hate myself"]
};

function detectCategory(text) {
  const t = text.toLowerCase();
  for (const [c, keys] of Object.entries(categories)) {
    if (keys.some(k => t.includes(k))) return c;
  }
  return "general_support";
}

/* ---------- RAG ---------- */
function cosine(a,b){
  let d=0,na=0,nb=0;
  for(let i=0;i<a.length;i++){
    d+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i];
  }
  return d/(Math.sqrt(na)*Math.sqrt(nb));
}

async function retrieveContext(text, category) {
  const q = await embedder(text,{pooling:"mean",normalize:true});
  return kbEmbeddings
    .filter(k=>k.category===category)
    .map(k=>({...k,score:cosine(q.data,k.embedding)}))
    .sort((a,b)=>b.score-a.score)
    .slice(0,2);
}

function buildPrompt(userText, ctx) {
  return `
You are a calm, empathetic mental wellness companion.
No advice, no diagnosis.

Context:
${ctx.map(c=>"- "+c.text).join("\n")}

User: "${userText}"

Respond warmly in 2–3 sentences.
`;
}

/* ---------- INIT ---------- */
async function initAI() {
  statusMessage.textContent = "Loading models locally...";
  embedder = await pipeline("feature-extraction","Xenova/all-MiniLM-L6-v2");
  generator = await pipeline("text-generation","Xenova/distilgpt2");

  for (const item of MENTAL_HEALTH_KB) {
    const e = await embedder(item.text,{pooling:"mean",normalize:true});
    kbEmbeddings.push({...item, embedding:e.data});
  }
  statusMessage.textContent = "AI ready (100% private)";
}
initAI();

/* ---------- UI ---------- */
function addMessage(sender,text){
  const div=document.createElement("div");
  div.className=`message ${sender}`;
  div.textContent=text;
  chatContainer.appendChild(div);
  chatContainer.scrollTop=chatContainer.scrollHeight;
}

/* ---------- MAIN ---------- */
async function handleMessage(){
  const text=userInput.value.trim();
  if(!text) return;

  addMessage("user",text);
  userInput.value="";

  let reply;
  if(containsCrisis(text)){
    reply=crisisResponse();
  } else {
    const cat=detectCategory(text);
    const ctx=await retrieveContext(text,cat);
    const prompt=buildPrompt(text,ctx);
    const out=await generator(prompt,{max_length:120,temperature:0.7});
    reply=out[0].generated_text.replace(prompt,"").trim();
  }

  addMessage("assistant",reply);
}

sendButton.onclick = handleMessage;
userInput.addEventListener("keydown",e=>{
  if(e.key==="Enter") handleMessage();
});
