"""
SFA First Aid RAG Chatbot — v2 Advanced
Singapore Red Cross | Standard First Aid Manual 2020
Streamlit deployment
"""
import re, time, io, warnings
import numpy as np
import streamlit as st
import torch
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from bert_score import score as bert_score_fn

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SFA RAG v2 — Advanced",
    page_icon="🏥",
    layout="wide",
)

# ─── Constants ─────────────────────────────────────────────────────────────────
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE           = torch.float16 if DEVICE == "cuda" else torch.float32
EMBEDDER_NAME   = "all-MiniLM-L6-v2"
RERANKER_NAME   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MODEL_NAME      = "google/flan-t5-large"
BERTSCORE_MODEL = "distilbert-base-uncased"

FEW_SHOT_EXAMPLES = """
--- EXAMPLES OF GOOD ANSWERS ---

Q: How do I treat a burn?
A: Treatment for Burns (SFA Manual):
Supplies needed: cool running water, clean non-fluffy dressing or cling wrap.
1. Remove the casualty from danger. Do NOT remove clothing stuck to the burn.
2. Cool with running water for at least 10 minutes. Never use ice, butter, or toothpaste.
3. Remove jewellery or watches near the burn area if not stuck to skin.
4. Cover loosely with a clean non-fluffy dressing or cling wrap.
5. Call 995 for severe burns: large area, face, hands, deep burns, airway involvement.
6. Monitor for signs of shock: pale, cold, rapid shallow breathing.

Q: What should I do if someone is unconscious?
A: Management of Unconscious Casualty (SFA Manual):
1. Ensure the scene is safe to approach.
2. Tap shoulders firmly and shout to check for response.
3. Call 995 (SCDF) immediately or direct a bystander to call.
4. Primary Survey: Danger, Response, Shout for help, Airway, Breathing, Circulation.
5. Not breathing normally: start CPR (30 compressions + 2 rescue breaths) and use AED.
6. Breathing normally: place in Recovery Position to keep airway clear.
7. Do not give anything by mouth. Do not move unnecessarily.

--- END EXAMPLES ---
"""

# ─── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedder():
    return SentenceTransformer(EMBEDDER_NAME, device=DEVICE)

@st.cache_resource(show_spinner="Loading cross-encoder reranker…")
def load_reranker():
    return CrossEncoder(RERANKER_NAME, device=DEVICE)

@st.cache_resource(show_spinner="Loading language model (flan-t5-large, ~780M)…")
def load_lm():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    if DEVICE == "cpu":
        mdl = mdl.to(DEVICE)
    mdl.eval()
    return tok, mdl

# ─── PDF Processing ────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = re.sub(r"Singapore Resuscitation and First Aid Council[^\n]*", "", text)
    text = re.sub(r"All rights reserved", "", text)
    text = re.sub(r"SRFAC\s*STANDARD FIRST AID", "", text)
    text = re.sub(r"Fig\.\s*\d+[^\n]*", "", text)
    text = re.sub(r"\(see figure[^)]*\)", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\x00-\x7F]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def detect_chapter(text: str) -> str:
    m = re.search(r"Chapter\s+(\d+)\s*[-]\s*([A-Za-z\s,/]+)", text)
    if m:
        return f"[Chapter {m.group(1)} - {m.group(2).strip()}]"
    for kw in ["CPR", "Wound", "Burn", "Fracture", "Shock", "Bleeding",
               "Unconscious", "Breathing", "Choking", "Asthma", "Stroke"]:
        if kw.lower() in text.lower():
            return f"[Topic: {kw}]"
    return ""

def extract_text(pdf_bytes: bytes) -> str:
    reader   = PdfReader(io.BytesIO(pdf_bytes))
    all_text = ""
    for page in reader.pages[5:]:
        raw = page.extract_text() or ""
        cleaned = clean_text(raw)
        if cleaned:
            chapter = detect_chapter(cleaned)
            prefix  = f"{chapter} " if chapter else ""
            all_text += f"\n{prefix}{cleaned}"
    return all_text

def build_index(all_text: str, embedder):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = [c.strip() for c in splitter.split_text(all_text) if len(c.strip()) > 80]
    embeddings = embedder.encode(
        chunks, batch_size=32, show_progress_bar=False, normalize_embeddings=True
    )
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    return chunks, index

# ─── RAG Bot ───────────────────────────────────────────────────────────────────
class AdvancedRAGBot_v7:
    GREETINGS = {"hi", "hello", "hey", "thanks", "thank you",
                 "ok", "okay", "bye", "good morning", "good afternoon"}

    def __init__(self, chunks, index, embedder, reranker, tokenizer, model):
        self.chunks    = chunks
        self.index     = index
        self.embedder  = embedder
        self.reranker  = reranker
        self.tokenizer = tokenizer
        self.model     = model
        self.k_init    = 20
        self.k_final   = 5
        self.max_length = 600
        self.min_length = 50
        self.history   = []

    def _rewrite_query(self, question: str) -> str:
        if not self.history:
            return question
        pronouns = ["it", "that", "this", "they", "those", "them"]
        if not any(p in question.lower().split() for p in pronouns):
            return question
        last_q = self.history[-1]["user"]
        return f"{last_q} — specifically: {question}"

    def retrieve(self, query: str):
        rewritten = self._rewrite_query(query)
        q_emb     = self.embedder.encode([rewritten], normalize_embeddings=True)
        _, I      = self.index.search(q_emb.astype("float32"), self.k_init)
        candidates = [self.chunks[i] for i in I[0] if i < len(self.chunks)]

        scores     = self.reranker.predict([[query, c] for c in candidates])
        ranked     = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        top        = [c for _, c in ranked[:self.k_final]]
        confidence = float(np.mean([1 / (1 + np.exp(-s)) for s, _ in ranked[:self.k_final]]))
        return top, confidence

    def build_prompt(self, context_chunks: list, question: str) -> str:
        ctx = "\n\n".join(
            f"[Passage {i}]\n{re.sub(r'\\[Topic:[^\\]]*\\]\\s*', '', c).strip()}"
            for i, c in enumerate(context_chunks, 1)
        )
        return (
            f"{FEW_SHOT_EXAMPLES}\n\n"
            f"===BEGIN SFA MANUAL CONTEXT===\n{ctx}\n===END SFA MANUAL CONTEXT===\n\n"
            f"Q: {question}\nA:"
        )

    def _format_answer(self, text: str) -> str:
        # Ensure numbered steps
        lines, out, step_n = text.split("\n"), [], 0
        for line in lines:
            stripped = line.strip()
            if stripped and not re.match(r"^\d+\.", stripped) and len(stripped) > 30:
                if re.match(r"^(Apply|Place|Press|Hold|Cover|Wrap|Cool|Remove|Call|Check|"
                            r"Perform|Stop|Avoid|Do not|Monitor|Wash|Rinse|Ensure)", stripped, re.I):
                    step_n += 1
                    out.append(f"{step_n}. {stripped}")
                    continue
            out.append(line)
        return "\n".join(out)

    def generate(self, question: str) -> dict:
        if question.lower().strip() in self.GREETINGS:
            return {
                "answer":    "Hello! I am the Singapore Red Cross SFA First Aid Assistant (v7.0 Advanced). How can I help?",
                "context":   [], "chunks_k": 0, "confidence": 1.0,
                "time_s":    0.0, "version": "v7.0",
            }

        t0          = time.time()
        top, conf   = self.retrieve(question)
        prompt      = self.build_prompt(top, question)

        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=1024, truncation=True
        ).to(DEVICE)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_length     = self.max_length,
                min_length     = self.min_length,
                num_beams      = 4,
                length_penalty = 2.0,
                early_stopping = True,
            )
        answer = self.tokenizer.decode(out[0], skip_special_tokens=True)
        answer = self._format_answer(answer)

        self.history.append({"user": question, "bot": answer[:200]})
        return {
            "answer":    answer,
            "context":   top,
            "chunks_k":  self.k_final,
            "confidence": conf,
            "time_s":    time.time() - t0,
            "version":   "v7.0",
        }

    def reset_memory(self):
        self.history = []

# ─── Metrics ───────────────────────────────────────────────────────────────────
_bs_cache = {}

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def compute_metrics(question: str, result: dict, embedder) -> dict:
    answer  = result["answer"]
    context = result["context"]
    k       = max(1, result["chunks_k"])
    m       = {}

    q_emb    = embedder.encode([question], normalize_embeddings=True)[0]
    ans_emb  = embedder.encode([answer],   normalize_embeddings=True)[0]
    clean_ctx = [re.sub(r"\[Topic:[^\]]*\]\s*", "", c).strip() for c in context] \
                if context else []
    ctx_embs = embedder.encode(clean_ctx, normalize_embeddings=True) \
               if clean_ctx else np.zeros((1, len(q_emb)))

    ctx_sims     = [cosine(q_emb, c) for c in ctx_embs]
    m["ctx_rel"] = float(np.mean(ctx_sims)) if ctx_sims else 0.0
    m["ans_rel"] = cosine(q_emb, ans_emb)

    sentences = [s.strip() for s in re.split(r"[.!?\n]", answer) if len(s.strip()) > 12]
    if sentences and context:
        sent_embs  = embedder.encode(sentences, normalize_embeddings=True)
        m["faith"] = float(np.mean(
            [max(cosine(se, ce) for ce in ctx_embs) for se in sent_embs]
        ))
    else:
        m["faith"] = 0.0

    relevant         = sum(1 for s in ctx_sims if s > 0.28)
    m["recall_k"]    = min(1.0, relevant / max(1, k))
    m["precision_k"] = relevant / k

    ctx_combined = " ".join(clean_ctx)[:2000] if context else ""
    cache_key    = (answer[:200], ctx_combined[:200])
    if cache_key in _bs_cache:
        m["bert_score"] = _bs_cache[cache_key]
    else:
        try:
            P, R, F1 = bert_score_fn(
                [answer], [ctx_combined],
                model_type=BERTSCORE_MODEL, lang="en", verbose=False, device=DEVICE
            )
            m["bert_score"] = float(F1[0])
        except Exception:
            m["bert_score"] = 0.0
        _bs_cache[cache_key] = m["bert_score"]

    m["lat_s"] = result.get("time_s", 0.0)
    return m

# ─── Metric rendering ──────────────────────────────────────────────────────────
def badge_color(v, good=0.75, mid=0.50):
    if v >= good: return "🟢"
    if v >= mid:  return "🟡"
    return "🔴"

def render_metrics(m: dict):
    st.markdown("**Evaluation Metrics**")
    cols  = st.columns(6)
    items = [
        ("Recall@K",     m["recall_k"]),
        ("Precision@K",  m["precision_k"]),
        ("CtxRel",       m["ctx_rel"]),
        ("Faithfulness", m["faith"]),
        ("AnsRel",       m["ans_rel"]),
        ("BERTScore",    m["bert_score"]),
    ]
    for col, (label, val) in zip(cols, items):
        col.metric(f"{badge_color(val)} {label}", f"{val:.3f}")
    st.caption(f"⏱ {m['lat_s']:.2f}s")

# ─── Main UI ───────────────────────────────────────────────────────────────────
def main():
    st.title("🏥 SFA First Aid RAG Chatbot — v2 Advanced")
    st.caption("Singapore Red Cross | Standard First Aid Manual 2020 | flan-t5-large · Bi-encoder + CrossEncoder reranker")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Setup")
        pdf_file = st.file_uploader(
            "Upload SFA Manual PDF",
            type="pdf",
            help="Upload SFA-Manual-Rev-1-2020_final.pdf",
        )
        st.divider()
        show_metrics = st.checkbox("Show evaluation metrics", value=False)
        show_context = st.checkbox("Show retrieved chunks",   value=False)
        if st.button("🗑️ Clear chat + memory"):
            st.session_state.messages = []
            if st.session_state.get("bot"):
                st.session_state.bot.reset_memory()
            st.rerun()
        st.divider()
        st.markdown("**Architecture**")
        st.markdown("- LM: `flan-t5-large` (780M)")
        st.markdown("- Embedder: `all-MiniLM-L6-v2`")
        st.markdown("- Reranker: `ms-marco-MiniLM-L-6-v2`")
        st.markdown("- Retrieval: top-20 → rerank → top-5")
        st.markdown("- Chunks: size=800, overlap=200")
        st.markdown("- Prompting: Few-shot with examples")
        st.markdown("- Memory: Query rewriting")

    # ── Session state ──────────────────────────────────────────────────────────
    if "messages"   not in st.session_state: st.session_state.messages   = []
    if "bot"        not in st.session_state: st.session_state.bot        = None
    if "pdf_loaded" not in st.session_state: st.session_state.pdf_loaded = False

    # ── PDF processing ─────────────────────────────────────────────────────────
    if pdf_file is not None and not st.session_state.pdf_loaded:
        embedder = load_embedder()
        reranker = load_reranker()
        with st.spinner("📄 Processing PDF and building FAISS index…"):
            all_text = extract_text(pdf_file.read())
            chunks, index = build_index(all_text, embedder)
        with st.spinner("🤖 Loading flan-t5-large (this may take a few minutes)…"):
            tokenizer, model = load_lm()
        st.session_state.bot = AdvancedRAGBot_v7(
            chunks, index, embedder, reranker, tokenizer, model
        )
        st.session_state.pdf_loaded = True
        st.session_state.messages   = []
        st.success(f"✅ Ready! {len(chunks)} chunks indexed.")

    # ── Chat display ───────────────────────────────────────────────────────────
    if not st.session_state.pdf_loaded:
        st.info("👆 Upload the SFA Manual PDF in the sidebar to begin.")
        return

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and show_metrics and "metrics" in msg:
                render_metrics(msg["metrics"])
            if msg["role"] == "assistant" and show_context and "context" in msg:
                with st.expander("📄 Retrieved Chunks"):
                    for i, chunk in enumerate(msg["context"], 1):
                        clean = re.sub(r"\[Topic:[^\]]*\]\s*", "", chunk).strip()
                        st.markdown(f"**[Chunk {i}]**\n\n{clean[:400]}{'...' if len(clean)>400 else ''}")

    # ── Input ──────────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask a first aid question…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result   = st.session_state.bot.generate(prompt)
                answer   = result["answer"]
                embedder = load_embedder()
                metrics  = compute_metrics(prompt, result, embedder) if show_metrics else None

            st.markdown(answer)
            if show_metrics and metrics:
                render_metrics(metrics)
            if show_context and result.get("context"):
                with st.expander("📄 Retrieved Chunks"):
                    for i, chunk in enumerate(result["context"], 1):
                        clean = re.sub(r"\[Topic:[^\]]*\]\s*", "", chunk).strip()
                        st.markdown(f"**[Chunk {i}]**\n\n{clean[:400]}{'...' if len(clean)>400 else ''}")

        msg_data = {
            "role":    "assistant",
            "content": answer,
            "context": result.get("context", []),
        }
        if metrics:
            msg_data["metrics"] = metrics
        st.session_state.messages.append(msg_data)

if __name__ == "__main__":
    main()
