"""
SFA First Aid RAG Chatbot — v1.0 Basic
Singapore Red Cross | Standard First Aid Manual 2020
Streamlit deployment
"""
import re, time, io, warnings
import numpy as np
import streamlit as st
import torch
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from bert_score import score as bert_score_fn

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SFA RAG v1 — Basic",
    page_icon="🩹",
    layout="wide",
)

# ─── Constants ─────────────────────────────────────────────────────────────────
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE         = torch.float16 if DEVICE == "cuda" else torch.float32
EMBEDDER_NAME = "all-MiniLM-L6-v2"
MODEL_NAME    = "google/flan-t5-base"
BERTSCORE_MODEL = "distilbert-base-uncased"

# ─── Model loading (cached across sessions) ────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedder():
    return SentenceTransformer(EMBEDDER_NAME, device=DEVICE)

@st.cache_resource(show_spinner="Loading language model (flan-t5-base)…")
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
def extract_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    all_text = ""
    for page in reader.pages[5:]:
        raw = page.extract_text() or ""
        raw = re.sub(r"\s+", " ", raw).strip()
        if raw:
            all_text += f"\n{raw}"
    return all_text

def build_index(all_text: str, embedder):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=50,
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
class BasicRAGBot_v1:
    def __init__(self, chunks, index, embedder, tokenizer, model):
        self.chunks    = chunks
        self.index     = index
        self.embedder  = embedder
        self.tokenizer = tokenizer
        self.model     = model
        self.k         = 2
        self.max_length = 150

    def retrieve(self, query):
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        _, I  = self.index.search(q_emb.astype("float32"), self.k)
        return [self.chunks[i] for i in I[0] if i < len(self.chunks)]

    def generate(self, question):
        t0         = time.time()
        ctx_chunks = self.retrieve(question)
        context    = "\n\n".join(ctx_chunks)
        prompt     = f"question: {question} context: {context}"

        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        ).to(DEVICE)

        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_length=self.max_length,
                num_beams=4, early_stopping=True,
            )
        answer = self.tokenizer.decode(out[0], skip_special_tokens=True)
        conf   = 0.5  # v1 has no confidence scoring
        return {
            "answer":    answer,
            "context":   ctx_chunks,
            "chunks_k":  self.k,
            "confidence": conf,
            "time_s":    time.time() - t0,
            "version":   "v1.0",
        }

# ─── Metrics ───────────────────────────────────────────────────────────────────
_bs_cache = {}

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def compute_metrics(question, result, embedder):
    answer  = result["answer"]
    context = result["context"]
    k       = max(1, result["chunks_k"])
    m       = {}

    q_emb    = embedder.encode([question],   normalize_embeddings=True)[0]
    ans_emb  = embedder.encode([answer],     normalize_embeddings=True)[0]
    ctx_embs = embedder.encode(context,      normalize_embeddings=True) if context \
               else np.zeros((1, len(q_emb)))

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

    ctx_combined = " ".join(context)[:2000] if context else ""
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

# ─── Metric badge helper ────────────────────────────────────────────────────────
def badge_color(v, good=0.75, mid=0.50):
    if v >= good: return "🟢"
    if v >= mid:  return "🟡"
    return "🔴"

def render_metrics(m):
    st.markdown("**Evaluation Metrics**")
    cols = st.columns(6)
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
    st.title("🩹 SFA First Aid RAG Chatbot — v1.0 Basic")
    st.caption("Singapore Red Cross | Standard First Aid Manual 2020 | flan-t5-base · Bi-encoder only")

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
        if st.button("🗑️ Clear chat history"):
            st.session_state.messages = []
            st.rerun()
        st.divider()
        st.markdown("**Architecture**")
        st.markdown("- LM: `flan-t5-base` (248M)")
        st.markdown("- Embedder: `all-MiniLM-L6-v2`")
        st.markdown("- Retrieval: Bi-encoder top-2")
        st.markdown("- Chunks: size=512, overlap=50")

    # ── Session state ──────────────────────────────────────────────────────────
    if "messages"   not in st.session_state: st.session_state.messages   = []
    if "bot"        not in st.session_state: st.session_state.bot        = None
    if "pdf_loaded" not in st.session_state: st.session_state.pdf_loaded = False

    # ── PDF processing ─────────────────────────────────────────────────────────
    if pdf_file is not None and not st.session_state.pdf_loaded:
        embedder = load_embedder()
        with st.spinner("📄 Processing PDF and building FAISS index…"):
            all_text = extract_text(pdf_file.read())
            chunks, index = build_index(all_text, embedder)
        with st.spinner("🤖 Loading flan-t5-base language model…"):
            tokenizer, model = load_lm()
        st.session_state.bot        = BasicRAGBot_v1(chunks, index, embedder, tokenizer, model)
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
                        st.markdown(f"**[Chunk {i}]**\n\n{chunk[:400]}{'...' if len(chunk)>400 else ''}")

    # ── Input ──────────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask a first aid question…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result  = st.session_state.bot.generate(prompt)
                answer  = result["answer"]
                embedder = load_embedder()
                metrics = compute_metrics(prompt, result, embedder) if show_metrics else None

            st.markdown(answer)
            if show_metrics and metrics:
                render_metrics(metrics)
            if show_context and result.get("context"):
                with st.expander("📄 Retrieved Chunks"):
                    for i, chunk in enumerate(result["context"], 1):
                        st.markdown(f"**[Chunk {i}]**\n\n{chunk[:400]}{'...' if len(chunk)>400 else ''}")

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
