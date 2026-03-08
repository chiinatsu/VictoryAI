# SFA First Aid RAG Chatbot — Streamlit Deployment

Singapore Red Cross | Standard First Aid Manual 2020

Three model versions, each as a standalone Streamlit app.

---

## Apps

| File | Version | Generator | Notes |
|---|---|---|---|
| `app_v1.py` | v1 Basic | `flan-t5-base` (248M) | Bi-encoder only, top-2 retrieval |
| `app_v2.py` | v2 Advanced | `flan-t5-large` (780M) | Bi-encoder + CrossEncoder reranker, few-shot prompting |
| `app_v3.py` | v3 Llama 3 | Llama 3.3-70B via Groq | 9-layer anti-hallucination, requires Groq API key |

---

## Prerequisites

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** `torch` installs the CPU version by default. For GPU support, install the correct CUDA build from [pytorch.org](https://pytorch.org) first.

> **Note:** The first run will download embedding/language models from HuggingFace (~300MB–1.5GB depending on version).

### 2. Groq API key (v3 only)
Get a **free** key at [console.groq.com](https://console.groq.com). You'll enter it in the app's sidebar.

---

## Running the Apps

Each app runs independently. Open a terminal in this folder and run:

**v1 — Basic RAG:**
```bash
streamlit run app_v1.py
```

**v2 — Advanced RAG:**
```bash
streamlit run app_v2.py
```

**v3 — Llama 3 / Groq:**
```bash
streamlit run app_v3.py
```

Each app opens at `http://localhost:8501` by default.

---

## Using the Apps

1. **Upload the PDF** — use the sidebar file uploader to upload `SFA-Manual-Rev-1-2020_final.pdf`
2. **For v3 only** — enter your Groq API key in the sidebar before uploading
3. Wait for the index to build (shown via a spinner — typically 30–90 seconds)
4. **Ask questions** in the chat box, e.g.:
   - *"How do I treat a burn injury?"*
   - *"What should I do if someone is choking?"*
   - *"What are the steps for performing CPR?"*

### Optional features (checkbox in sidebar)
- **Show evaluation metrics** — displays Recall@K, Precision@K, CtxRel, Faithfulness, AnsRel, BERTScore (and Hallucination score for v3)
- **Show retrieved chunks** — shows the raw passages used to generate the answer

---

## Hardware notes

| Version | Recommended | Minimum |
|---|---|---|
| v1 | CPU is fine | Any |
| v2 | GPU strongly recommended | CPU (slow, ~60s/response) |
| v3 | CPU is fine (generation is on Groq servers) | Any with internet access |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ImportError: langchain_text_splitters` | Run `pip install langchain-text-splitters` |
| v3: `Groq API error: 401` | Check your API key — it must start with `gsk_` |
| v3: `429 rate limit` | The app auto-retries and falls back to `llama-3.1-8b-instant` automatically |
| Slow response on v2 | Use GPU or reduce `k_final` in the code |
| PDF not loading | Ensure you're uploading the correct SFA Manual PDF |
