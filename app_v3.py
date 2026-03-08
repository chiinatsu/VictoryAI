"""
SFA First Aid RAG Chatbot — v3 Llama 3 Edition
Singapore Red Cross | Standard First Aid Manual 2020
Streamlit deployment
"""
import re, time, io, math, warnings, difflib
import numpy as np
import streamlit as st
import torch
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from bert_score import score as bert_score_fn
from groq import Groq

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SFA RAG v3 — Llama 3",
    page_icon="🧠",
    layout="wide",
)

# ─── Constants ─────────────────────────────────────────────────────────────────
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDER_MODEL  = "multi-qa-mpnet-base-dot-v1"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-12-v2"
GROQ_MODEL      = "llama-3.3-70b-versatile"
FALLBACK_MODEL  = "llama-3.1-8b-instant"
BERTSCORE_MODEL = "distilbert-base-uncased"

# ─── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading bi-encoder (multi-qa-mpnet-base-dot-v1)…")
def load_embedder():
    return SentenceTransformer(EMBEDDER_MODEL, device=DEVICE)

@st.cache_resource(show_spinner="Loading cross-encoder reranker (ms-marco-MiniLM-L-12)…")
def load_reranker():
    return CrossEncoder(RERANKER_MODEL, device=DEVICE)

# ─── PDF Processing ────────────────────────────────────────────────────────────
CHAPTER_KEYWORDS = {
    "1": ["first aid","primary survey","secondary survey","universal precaution"],
    "2": ["unconscious","recovery position","head injury","spinal","fainting",
          "fits","stroke","blood sugar","heat disorder"],
    "3": ["airway","choking","asthma","hyperventilation","fumes","allergic"],
    "4": ["shock","bleeding","wound","bandage"],
    "5": ["fracture","dislocation","sprain","soft tissue","immobilis"],
    "6": ["burn","scald","chemical burn","electrical burn"],
    "7": ["eye","epistaxis","nose bleed","poison","transport"],
    "8": ["cpr","aed","defibrillat","cardiac arrest","chest compression"],
}

def clean_text(text: str) -> str:
    text = re.sub(r"Singapore Resuscitation[^\n]*", "", text)
    text = re.sub(r"SINGAPORE RED CROSS ACADEMY[^\n]*", "", text)
    text = re.sub(r"SRFAC\s*STANDARD FIRST AID", "", text)
    text = re.sub(r"All rights reserved", "", text)
    text = re.sub(r"Fig\.\s*\d+[^\n]*", "", text)
    text = re.sub(r"\(see figure[^)]*\)", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\x00-\x7F]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def detect_chapter(text: str) -> str:
    m = re.search(r"Chapter\s*(\d+)\s*[-]\s*([A-Za-z ,/]+)", text)
    if m:
        return f"[Chapter {m.group(1).strip()} - {m.group(2).strip()}]"
    t_lower = text.lower()
    for ch_num, kws in CHAPTER_KEYWORDS.items():
        if any(k in t_lower for k in kws):
            return f"[Chapter {ch_num}]"
    return ""

def extract_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    all_text, current_chapter = "", ""
    for page in reader.pages[5:]:
        raw     = page.extract_text() or ""
        cleaned = clean_text(raw)
        if not cleaned: continue
        chapter = detect_chapter(cleaned)
        if chapter: current_chapter = chapter
        prefix   = f"{current_chapter} " if current_chapter else ""
        all_text += f"\n{prefix}{cleaned}"
    return all_text

def build_index(all_text: str, embedder):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )
    chunks = [c.strip() for c in splitter.split_text(all_text) if len(c.strip()) > 60]
    embeddings = embedder.encode(
        chunks, batch_size=32, show_progress_bar=False, normalize_embeddings=True
    )
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    return chunks, index

# ─── Groq helpers ──────────────────────────────────────────────────────────────
def _parse_retry_after(error_str: str) -> float:
    m = re.search(r"try again in (\d+)m([\d.]+)s", str(error_str))
    if m:
        return int(m.group(1)) * 60 + float(m.group(2)) + 2
    m = re.search(r"try again in ([\d.]+)s", str(error_str))
    if m:
        return float(m.group(1)) + 2
    return 15

def call_llama(api_key: str, system_prompt: str, user_prompt: str,
               model=None, temperature=0.0, max_tokens=500, top_p=0.05,
               allow_fallback=True) -> str:
    client  = Groq(api_key=api_key)
    primary = model or GROQ_MODEL
    models  = [primary]
    if allow_fallback and primary != FALLBACK_MODEL:
        models.append(FALLBACK_MODEL)

    last_err = None
    for m in models:
        for retry in range(2):
            try:
                resp = client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop=["===END===", "\nNote:", "\nDisclaimer:", "\nIMPORTANT NOTE:"],
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                last_err = e
                err_str  = str(e)
                if "429" in err_str or "rate_limit" in err_str.lower():
                    wait = _parse_retry_after(err_str)
                    if retry == 0:
                        time.sleep(min(wait, 90))
                        continue
                    else:
                        break
                else:
                    raise e
    raise last_err

# ─── Query helpers ─────────────────────────────────────────────────────────────
COMMON_WORDS = {
    "the","and","for","are","but","not","you","all","can","was","one","our",
    "out","day","get","has","him","his","how","its","may","now","see","two",
    "way","who","did","let","put","say","she","too","use","that","this","with",
    "have","from","they","will","been","when","what","your","each","time","then",
    "them","into","just","like","long","make","many","more","only","over","such",
    "take","than","well","also","back","call","came","come","does","down","even",
    "find","give","here","keep","know","last","left","life","live","look","made",
    "move","need","open","part","stop","turn","very","want","work","year","after",
    "again","could","every","first","going","great","other","place","right",
    "small","still","these","think","three","under","until","which","while",
    "where","there","about",
}
MEDICAL_VOCAB = [
    "burn","scald","wound","bleed","bleeding","fracture","choking","choke",
    "cpr","cardiac","arrest","unconscious","fainting","shock","asthma","sprain",
    "stroke","seizure","eye","nose","poison","bite","sting","bandage","bruise",
    "cut","dressing","tourniquet","aed","airway","breathing","stab","puncture",
    "chemical","electrical","allergy","anaphylaxis","heat","hypothermia","drown",
    "swelling","broken","injury","chest","compression","rescue","recovery",
    "position","defibrillator","splint","sling","pressure","nosebleed","faint",
    "concussion","spinal","laceration","abrasion","contusion","amputation",
    "epistaxis","hypoglycaemia","hyperventilation","resuscitation","haemorrhage",
    "immobilise","immobilize","dislocation","avulsion","tourniquet","triage",
]
SYNONYM_MAP = {
    r"\bheart attack\b"                      : "cardiac arrest CPR chest compressions AED",
    r"\bcardiac arrest\b"                    : "cardiac arrest CPR AED chest compressions",
    r"\bcpr\b"                               : "CPR chest compressions rescue breaths cardiac",
    r"\bchoke\b|\bchoking\b"                 : "choking foreign body airway obstruction back blows",
    r"\bfaint\b|\bfainted\b|\bfainting\b"    : "fainting unconscious recovery position",
    r"\bseizure\b|\bconvuls|\bfits\b"        : "fits seizure unconscious recovery position",
    r"\bsprain\b|\bstrain\b"                 : "sprain soft tissue injury RICE immobilise",
    r"\bstroke\b"                            : "stroke FAST face arm speech",
    r"\blow sugar\b|\bhypo\b"                : "low blood sugar hypoglycaemia glucose",
    r"\bnosebleed\b|\bepistaxis\b"           : "epistaxis nose bleed pressure",
    r"\belectric\b|\belectrocut"             : "electrocution electrical burn",
    r"\bstab\b|\bpuncture\b|\bknife\b"       : "stab wound penetrating bleeding pressure",
    r"\bburn\b|\bscald\b"                    : "burn scald cool running water dressing",
    r"\bbleed\b|\bbleeding\b|\bhaemorrhage\b": "bleeding wound direct pressure bandage",
    r"\bbroken bone\b|\bfracture\b"          : "fracture broken bone immobilise splint",
    r"\ballergic\b|\banaphylaxis\b"          : "allergic reaction anaphylaxis",
    r"\bpoisoning\b|\bpoison\b"              : "poisoning toxic substance call 995",
    r"\bdrown\b|\bnear.drown"                : "drowning water rescue recovery position",
    r"\brice\b"                              : "RICE rest ice compression elevation sprain",
    r"\bdrsabc\b|\bprimary survey\b"         : "DRSABC danger response shout airway breathing",
    r"\baed\b|\bdefibrillat"                 : "AED defibrillator cardiac arrest CPR",
}
CHAPTER_TOPIC_MAP = {
    "burn":"6","scald":"6","bleed":"4","wound":"4","shock":"4","stab":"4",
    "haemorrhage":"4","tourniquet":"4","fracture":"5","sprain":"5","disloc":"5",
    "splint":"5","sling":"5","rice":"5","chok":"3","airway":"3","asthma":"3",
    "hypervent":"3","fumes":"3","allerg":"3","cpr":"8","cardiac":"8","aed":"8",
    "defibril":"8","compress":"8","unconsci":"2","faint":"2","stroke":"2",
    "seizure":"2","fits":"2","heat":"2","spinal":"2","recovery":"2","head injur":"2",
    "eye":"7","poison":"7","nose":"7","epistaxis":"7","nosebleed":"7",
    "drsabc":"1","primary survey":"1","ppe":"1","universal precaution":"1",
}

def correct_typos(query: str) -> str:
    words, corrected = query.split(), []
    for word in words:
        w = re.sub(r"[^a-zA-Z]", "", word).lower()
        if len(w) < 4 or w in MEDICAL_VOCAB or w in COMMON_WORDS:
            corrected.append(word); continue
        m = difflib.get_close_matches(w, MEDICAL_VOCAB, n=1, cutoff=0.82)
        corrected.append(m[0] if m else word)
    return " ".join(corrected)

def expand_query(query: str) -> str:
    expanded = query
    for pat, exp in SYNONYM_MAP.items():
        if re.search(pat, query, re.IGNORECASE):
            expanded += " " + exp
    return expanded

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def chapter_boost(query: str, chunk: str, base_score: float, boost=0.08) -> float:
    q_lower = query.lower()
    for kw, ch_num in CHAPTER_TOPIC_MAP.items():
        if kw in q_lower and f"[Chapter {ch_num}" in chunk:
            return min(base_score + boost, 1.0)
    return base_score

def mmr_select(query_emb, candidate_embs, candidates, k=7, lambda_=0.65):
    selected_idx, selected_embs = [], []
    rel_scores = [float(np.dot(query_emb, e)) for e in candidate_embs]
    for _ in range(min(k, len(candidates))):
        best_score, best_idx = -999, -1
        for i, (emb, rel) in enumerate(zip(candidate_embs, rel_scores)):
            if i in selected_idx: continue
            sim_sel = max((float(np.dot(emb, se)) for se in selected_embs), default=0.0)
            mmr_s   = lambda_ * rel - (1 - lambda_) * sim_sel
            if mmr_s > best_score:
                best_score, best_idx = mmr_s, i
        if best_idx >= 0:
            selected_idx.append(best_idx)
            selected_embs.append(candidate_embs[best_idx])
    return [candidates[i] for i in selected_idx]

FORMAT_SYSTEM = (
    "You are a first-aid guide formatter for the Singapore Red Cross.\n"
    "You receive verbatim sentences already extracted from the SFA Manual.\n"
    "Organise ONLY these sentences into a first-aid guide.\n\n"
    "RULES — no exceptions:\n"
    "1. Begin: [TOPIC] - First Aid (SFA Manual 2020):\n"
    "2. Sections: Supplies / Steps (numbered) / Warnings / Emergency.\n"
    "3. Use ONLY the provided sentences. Add ZERO new content.\n"
    "4. Do not merge, split, or rephrase any sentence.\n"
    "5. If a section has no applicable sentences, omit that section."
)
ULTRA_STRICT_SYSTEM = (
    "You are a Singapore Red Cross First Aid assistant.\n"
    "Use ONLY text from the provided context passages. No paraphrasing.\n"
    "Begin with '[TOPIC] - First Aid (SFA Manual 2020):'\n"
    "Supplies / numbered Steps / Warnings / Emergency."
)

def _build_fallback_prompt(context_chunks, question):
    parts = [f"[Passage {i}]\n{chunk}" for i, chunk in enumerate(context_chunks, 1)]
    return (
        "===BEGIN SFA MANUAL CONTEXT===\n"
        + "\n\n".join(parts)
        + "\n===END SFA MANUAL CONTEXT===\n\n"
        f"Question: {question}\n\n"
        "Answer using ONLY these passages. Copy phrases verbatim. "
        "Begin with the topic keyword."
    )

# ─── RAG Bot ───────────────────────────────────────────────────────────────────
class SFALlamaBot:
    GREETINGS = {
        "hi","hello","hey","thanks","thank you","ok","okay",
        "bye","good morning","good afternoon","who are you",
    }
    ANCHOR_THRESHOLD = 0.70
    ANCHOR_MIN_SIM   = 0.18
    STEP_THRESHOLD   = 0.65

    def __init__(self, chunks, index, embedder, reranker, api_key):
        self.chunks              = chunks
        self.index               = index
        self.embedder            = embedder
        self.reranker            = reranker
        self.api_key             = api_key
        self.version             = "v3.0"
        self.label               = f"v3.0 Llama 3 RAG 9-Layer ({GROQ_MODEL})"
        self.k_init              = 60
        self.k_final             = 7
        self.h_retry_threshold   = 0.20
        self.h_extract_threshold = 0.28
        self.confidence_gate     = 0.20
        self.history             = []

    def _retrieve(self, question, extra_query=None):
        expanded_q = expand_query(question)
        seen, candidates = set(), []
        for q in ([expanded_q, question] + ([extra_query] if extra_query else [])):
            q_emb = self.embedder.encode([q], normalize_embeddings=True)
            _, I  = self.index.search(q_emb.astype("float32"), min(self.k_init, len(self.chunks)))
            for idx in I[0]:
                if idx < len(self.chunks) and idx not in seen:
                    seen.add(idx); candidates.append(self.chunks[idx])

        raw_scores = self.reranker.predict([[question, c] for c in candidates])
        boosted    = sorted(
            [(chapter_boost(question, c, float(s)), c)
             for s, c in zip(raw_scores, candidates)],
            key=lambda x: x[0], reverse=True
        )
        top25    = [c for _, c in boosted[:25]]
        top25_e  = self.embedder.encode(top25, normalize_embeddings=True)
        q_emb_f  = self.embedder.encode([question], normalize_embeddings=True)[0]
        final    = mmr_select(q_emb_f, top25_e, top25, k=self.k_final)
        top_sigs   = [sigmoid(s) for s, _ in boosted[:self.k_final]]
        confidence = 0.60 * float(np.mean(top_sigs)) + 0.40 * max(top_sigs, default=0.0)
        return final, confidence, q_emb_f

    def _local_extract(self, question, context_chunks, q_emb, n_action=10, n_info=3):
        ACTION_RE = re.compile(
            r"\b(apply|place|press|hold|cover|wrap|cool|flush|remove|keep|"
            r"lay|sit|raise|lower|tilt|turn|roll|support|secure|bandage|"
            r"immobilis|elevate|compress|call|check|perform|stop|avoid|"
            r"do not|do NOT|loosen|monitor|reassure|wash|rinse|irrigate|"
            r"seal|pad|ensure|position|maintain|administer)\b", re.IGNORECASE
        )
        WARN_RE   = re.compile(r"\b(do not|do NOT|never|avoid|caution|warning|danger|must not|should not)\b", re.IGNORECASE)
        SUPPLY_RE = re.compile(r"\b(bandage|dressing|gloves|gauze|splint|sling|ice|towel|cloth|water|soap|blanket|ppe|mask|equipment|material|antiseptic|plaster)\b", re.IGNORECASE)
        EMERG_RE  = re.compile(r"\b(995|SCDF|ambulance|hospital|emergency|dial|call for help)\b", re.IGNORECASE)

        all_sents = []
        for chunk in context_chunks:
            clean = re.sub(r"\[Chapter[^\]]*\]\s*", "", chunk)
            for part in re.split(r"(?<=[.!?])\s+|\n", clean):
                p = part.strip()
                if len(p) > 20:
                    all_sents.append(p)
        if not all_sents:
            return None

        s_embs = self.embedder.encode(all_sents, normalize_embeddings=True)
        scored = [(s, float(np.dot(q_emb, e))) for s, e in zip(all_sents, s_embs)]
        scored.sort(key=lambda x: x[1], reverse=True)

        steps    = [(s, v) for s, v in scored if ACTION_RE.search(s) and not WARN_RE.search(s)][:n_action]
        warnings = [(s, v) for s, v in scored if WARN_RE.search(s)][:4]
        supplies = [(s, v) for s, v in scored if SUPPLY_RE.search(s)][:4]
        emergency= [(s, v) for s, v in scored if EMERG_RE.search(s)][:2]
        info     = [(s, v) for s, v in scored
                    if not ACTION_RE.search(s) and not WARN_RE.search(s)
                    and not SUPPLY_RE.search(s) and not EMERG_RE.search(s)][:n_info]
        return {
            "steps": [s for s, _ in steps], "warnings": [s for s, _ in warnings],
            "supplies": [s for s, _ in supplies], "emergency": [s for s, _ in emergency],
            "info": [s for s, _ in info],
            "all": [s for s, _ in (steps + warnings + supplies + emergency + info)],
        }

    def _local_to_answer(self, extracted, question):
        topic = question.strip().rstrip("?").title()
        lines = [f"{topic} - First Aid (SFA Manual 2020):"]
        if extracted.get("supplies"):
            lines += ["", "Supplies:"] + [f"  - {s}" for s in extracted["supplies"]]
        if extracted.get("steps"):
            lines += ["", "Steps:"] + [f"  {i}. {s}" for i, s in enumerate(extracted["steps"], 1)]
        elif extracted.get("info"):
            lines += ["", "Steps:"] + [f"  {i}. {s}" for i, s in enumerate(extracted["info"], 1)]
        if extracted.get("warnings"):
            lines += ["", "Warnings:"] + [f"  - {s}" for s in extracted["warnings"]]
        if extracted.get("emergency"):
            lines += ["", "Emergency:"] + [f"  - {s}" for s in extracted["emergency"]]
        lines += ["", "[Extracted verbatim from SFA Manual 2020]"]
        return "\n".join(lines)

    def _check_grounding(self, answer, context_chunks):
        THRESHOLD = 0.22
        skip_re   = re.compile(r"^(\[|Supplies:|Steps:|Warnings:|Emergency:|This topic|\d+\.\s{0,2}$|[-*]\s)", re.IGNORECASE)
        sentences = [
            s.strip() for s in re.split(r"[.!\n]", answer)
            if len(s.strip()) > 20 and not skip_re.match(s.strip())
        ]
        if not sentences or not context_chunks:
            return 0.0, []
        sent_embs = self.embedder.encode(sentences, normalize_embeddings=True)
        ctx_embs  = self.embedder.encode(context_chunks, normalize_embeddings=True)
        flagged, n_bad = [], 0
        for sent, s_emb in zip(sentences, sent_embs):
            max_sim = max(float(np.dot(s_emb, c_emb)) for c_emb in ctx_embs)
            if max_sim < THRESHOLD:
                flagged.append((sent, round(max_sim, 3))); n_bad += 1
        return (n_bad / len(sentences)), flagged

    def _build_ctx_sent_pool(self, context_chunks):
        ctx_sents = []
        for chunk in context_chunks:
            clean = re.sub(r"\[Chapter[^\]]*\]\s*", "", chunk)
            for part in re.split(r"(?<=[.!?])\s+|\n", clean):
                p = part.strip()
                if len(p) > 18:
                    ctx_sents.append(p)
        return ctx_sents

    def _anchor_sentences(self, answer, ctx_sents, ctx_sent_embs):
        KEEP_RE = re.compile(r"^(\[|Supplies:|Steps:|Warnings:|Emergency:|This topic|NOTE:|[-*]\s)", re.IGNORECASE)
        lines, anchored = answer.split("\n"), []
        for line in lines:
            stripped = line.strip()
            if not stripped or KEEP_RE.match(stripped) or len(stripped) < 22:
                anchored.append(line); continue
            content  = re.sub(r"^\d+\.\s*", "", stripped)
            line_emb = self.embedder.encode([content], normalize_embeddings=True)[0]
            sims     = [float(np.dot(line_emb, ce)) for ce in ctx_sent_embs]
            best_sim = max(sims); best_idx = int(np.argmax(sims))
            if best_sim < self.ANCHOR_THRESHOLD and best_sim >= self.ANCHOR_MIN_SIM:
                indent = line[: len(line) - len(line.lstrip())]
                prefix = re.match(r"^(\d+\.\s*)", stripped)
                pstr   = prefix.group(1) if prefix else ""
                anchored.append(f"{indent}{pstr}{ctx_sents[best_idx]}")
            else:
                anchored.append(line)
        return "\n".join(anchored)

    def _verify_steps(self, answer, ctx_sents, ctx_sent_embs):
        lines, verified = answer.split("\n"), []
        step_re = re.compile(r"^(\d+)\.\s+(.+)$")
        for line in lines:
            m = step_re.match(line.strip())
            if not m:
                verified.append(line); continue
            step_num, step_text = m.group(1), m.group(2).strip()
            if len(step_text) < 15:
                verified.append(line); continue
            step_emb = self.embedder.encode([step_text], normalize_embeddings=True)[0]
            sims     = [float(np.dot(step_emb, ce)) for ce in ctx_sent_embs]
            best_sim = max(sims); best_idx = int(np.argmax(sims))
            indent   = line[: len(line) - len(line.lstrip())]
            if best_sim < self.STEP_THRESHOLD and best_sim >= self.ANCHOR_MIN_SIM:
                verified.append(f"{indent}{step_num}. {ctx_sents[best_idx]}")
            else:
                verified.append(line)
        return "\n".join(verified)

    def generate(self, question: str) -> dict:
        if question.lower().strip() in self.GREETINGS:
            return {
                "answer": f"Hello! I am the Singapore Red Cross First Aid Assistant ({self.label}).",
                "context": [], "chunks_k": 0, "confidence": 1.0,
                "h_score": 0.0, "layer": "greeting",
                "time_s": 0.0, "tokens_used": 0, "version": self.version,
            }

        t0 = time.time()
        question_clean = correct_typos(question)
        top, confidence, q_emb = self._retrieve(question_clean)

        if confidence < self.confidence_gate or not top:
            answer = ("I could not find specific information about this topic in the "
                      "SFA Manual. Please consult a certified first aider or call 995.")
            return {
                "answer": answer, "context": top, "chunks_k": len(top),
                "confidence": confidence, "h_score": 1.0, "layer": "gate",
                "time_s": time.time() - t0, "tokens_used": 0, "version": self.version,
            }

        extracted   = self._local_extract(question_clean, top, q_emb)
        tokens_used = 0
        layer_used  = "L2a-local"
        answer      = None

        if extracted and extracted["steps"]:
            sentences_text = "\n".join(f"{i}. {s}" for i, s in enumerate(extracted["all"][:18], 1))
            topic    = question_clean.strip().rstrip("?").title()
            fmt_prompt = (
                f"Question topic: {topic}\n\n"
                "Extracted sentences from SFA Manual (verbatim):\n"
                f"{sentences_text}\n\n"
                f"Organise ONLY these sentences into a '{topic} - First Aid' guide."
            )
            try:
                answer = call_llama(
                    self.api_key, FORMAT_SYSTEM, fmt_prompt,
                    max_tokens=500, temperature=0.0, top_p=0.05
                )
                tokens_used = len(fmt_prompt.split()) + len(answer.split())
                layer_used  = "L2a+2b"
            except Exception:
                answer      = self._local_to_answer(extracted, question_clean)
                layer_used  = "L2a-only"
        else:
            try:
                answer = call_llama(
                    self.api_key, ULTRA_STRICT_SYSTEM,
                    _build_fallback_prompt(top, question_clean),
                    max_tokens=600, temperature=0.0, top_p=0.05
                )
                tokens_used = len(answer.split())
                layer_used  = "L2-single"
            except Exception:
                answer     = self._local_to_answer(extracted or {}, question_clean)
                layer_used = "L2a-only"

        h_score, flagged = self._check_grounding(answer, top)

        if h_score > self.h_retry_threshold and layer_used != "L2a-only":
            layer_used = "L5-retry"
            try:
                retry_ans = call_llama(
                    self.api_key, ULTRA_STRICT_SYSTEM,
                    _build_fallback_prompt(top, question_clean),
                    max_tokens=600, temperature=0.0, top_p=0.05
                )
                h_retry, f_retry = self._check_grounding(retry_ans, top)
                tokens_used += len(retry_ans.split())
                if h_retry <= h_score:
                    answer, h_score, flagged = retry_ans, h_retry, f_retry
            except Exception:
                pass

        if h_score > self.h_extract_threshold:
            layer_used = "L6-extract"
            if extracted:
                answer  = self._local_to_answer(extracted, question_clean)
                h_score = 0.0
                flagged = []

        if layer_used not in ("gate",):
            ctx_sents = self._build_ctx_sent_pool(top)
            if ctx_sents:
                ctx_sent_embs = self.embedder.encode(ctx_sents, normalize_embeddings=True)
                answer        = self._anchor_sentences(answer, ctx_sents, ctx_sent_embs)
                answer        = self._verify_steps(answer, ctx_sents, ctx_sent_embs)
            h_score, flagged = self._check_grounding(answer, top)

        self.history.append({"user": question, "bot": answer})
        return {
            "answer":      answer, "context": top,
            "chunks_k":    len(top), "confidence": confidence,
            "h_score":     h_score, "flagged": flagged, "layer": layer_used,
            "time_s":      time.time() - t0,
            "tokens_used": tokens_used,
            "version":     self.version,
        }

    def reset_memory(self):
        self.history = []

# ─── Metrics ───────────────────────────────────────────────────────────────────
_bs_cache       = {}
_CHAPTER_TAG_RE = re.compile(r"\[Chapter[^\]]*\]\s*")
_SKIP_FAITH_RE  = re.compile(r"^(\[|\s*[-*]\s*$|\d+\.\s*$)", re.IGNORECASE)

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def _extract_steps_text(answer):
    step_re = re.compile(r"^\s*\d+\.\s+(.+)$", re.MULTILINE)
    steps   = step_re.findall(answer)
    if len(steps) >= 2:
        return " ".join(steps)
    lines = [l.strip() for l in answer.split("\n")
             if l.strip()
             and not re.match(r"^(Supplies:|Steps:|Warnings:|Emergency:|NOTE:|\[)", l.strip(), re.I)
             and not re.match(r"^\d+\.\s*$", l.strip())]
    return " ".join(lines).strip() or answer

def compute_metrics(question: str, result: dict, embedder) -> dict:
    answer  = result["answer"]
    context = result["context"]
    k       = max(1, result["chunks_k"])
    m       = {}

    q_emb      = embedder.encode([question], normalize_embeddings=True)[0]
    steps_text = _extract_steps_text(answer)
    ans_emb    = embedder.encode([steps_text], normalize_embeddings=True)[0]
    clean_ctx  = [_CHAPTER_TAG_RE.sub("", c).strip() for c in context] if context else []
    ctx_embs   = embedder.encode(clean_ctx, normalize_embeddings=True) \
                 if clean_ctx else np.zeros((1, len(q_emb)))

    ctx_sims     = [cosine(q_emb, c) for c in ctx_embs]
    m["ctx_rel"] = float(np.mean(ctx_sims)) if ctx_sims else 0.0
    m["ans_rel"] = cosine(q_emb, ans_emb)

    sentences = [
        s.strip() for s in re.split(r"[.!?\n]", answer)
        if len(s.strip()) > 30 and not _SKIP_FAITH_RE.match(s.strip())
    ]
    if sentences and context:
        sent_embs  = embedder.encode(sentences, normalize_embeddings=True)
        m["faith"] = float(np.mean(
            [max(cosine(se, ce) for ce in ctx_embs) for se in sent_embs]
        ))
    else:
        m["faith"] = 0.0

    relevant         = sum(1 for s in ctx_sims if s > 0.25)
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

    m["lat_s"]   = result.get("time_s", 0.0)
    m["h_score"] = result.get("h_score", 0.0)
    return m

# ─── UI helpers ────────────────────────────────────────────────────────────────
def badge_color(v, good=0.75, mid=0.50):
    if v >= good: return "🟢"
    if v >= mid:  return "🟡"
    return "🔴"

def halluc_color(v):
    if v < 0.10: return "🟢"
    if v < 0.30: return "🟡"
    return "🔴"

LAYER_COLORS = {
    "L2a+2b":    "#2563eb", "L2a-only":  "#7c3aed",
    "L5-retry":  "#d97706", "L6-extract":"#dc2626",
    "greeting":  "#6b7280", "gate":      "#9ca3af",
}

def render_metrics(m: dict, layer: str = "", conf: float = 0.0, tokens: int = 0):
    st.markdown("**Evaluation Metrics**")
    cols  = st.columns(7)
    items = [
        ("Recall@K",     m["recall_k"],    False),
        ("Precision@K",  m["precision_k"], False),
        ("CtxRel",       m["ctx_rel"],     False),
        ("Faithfulness", m["faith"],       False),
        ("AnsRel",       m["ans_rel"],     False),
        ("BERTScore",    m["bert_score"],  False),
        ("Halluc↓",      m["h_score"],     True),
    ]
    for col, (label, val, inv) in zip(cols, items):
        icon = halluc_color(val) if inv else badge_color(val)
        col.metric(f"{icon} {label}", f"{val:.3f}")
    color = LAYER_COLORS.get(layer, "#374151")
    st.caption(
        f"⏱ {m['lat_s']:.2f}s  |  conf={conf*100:.0f}%  |  ~{tokens} tokens  |  "
        f'<span style="background:{color};color:white;padding:1px 6px;border-radius:8px;font-size:0.8em">'
        f"layer: {layer}</span>",
        unsafe_allow_html=True,
    )

# ─── Main UI ───────────────────────────────────────────────────────────────────
def main():
    st.title("🧠 SFA First Aid RAG Chatbot — v3 Llama 3 Edition")
    st.caption("Singapore Red Cross | Standard First Aid Manual 2020 | Llama 3.3-70B via Groq · 9-Layer Anti-Hallucination")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Setup")
        groq_key = st.secrets.get("GROQ_API_KEY", "") or st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_…",
            help="Get a free key at console.groq.com",
        )
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
        st.markdown("- Generator: Llama 3.3-70B (Groq)")
        st.markdown("- Embedder: `multi-qa-mpnet-base-dot-v1`")
        st.markdown("- Reranker: `ms-marco-MiniLM-L-12-v2`")
        st.markdown("- Retrieval: k=60 → rerank → MMR top-7")
        st.markdown("- Chunks: size=700, overlap=200")
        st.markdown("- 9-layer anti-hallucination defense")
        st.markdown("- Auto-retry + fallback model (8B)")

    # ── Session state ──────────────────────────────────────────────────────────
    if "messages"   not in st.session_state: st.session_state.messages   = []
    if "bot"        not in st.session_state: st.session_state.bot        = None
    if "pdf_loaded" not in st.session_state: st.session_state.pdf_loaded = False

    # ── Validation ─────────────────────────────────────────────────────────────
    if not groq_key:
        st.info("👆 Enter your Groq API key in the sidebar to begin.")
        return
    if not groq_key.startswith("gsk_"):
        st.warning("⚠️ Groq API keys should start with `gsk_`.")
        return

    # ── PDF processing ─────────────────────────────────────────────────────────
    if pdf_file is not None and not st.session_state.pdf_loaded:
        embedder = load_embedder()
        reranker = load_reranker()
        with st.spinner("📄 Processing PDF and building FAISS index…"):
            all_text = extract_text(pdf_file.read())
            chunks, index = build_index(all_text, embedder)
        # Test Groq connection
        with st.spinner("🔑 Testing Groq API connection…"):
            try:
                call_llama(groq_key, "Test.", "Say OK.", max_tokens=5)
            except Exception as e:
                st.error(f"Groq API error: {e}")
                return
        st.session_state.bot = SFALlamaBot(chunks, index, embedder, reranker, groq_key)
        st.session_state.pdf_loaded = True
        st.session_state.messages   = []
        st.success(f"✅ Ready! {len(chunks)} chunks indexed. Groq API connected.")
    elif st.session_state.pdf_loaded and st.session_state.bot:
        # Update API key if changed
        st.session_state.bot.api_key = groq_key

    # ── Chat display ───────────────────────────────────────────────────────────
    if not st.session_state.pdf_loaded:
        st.info("👆 Upload the SFA Manual PDF in the sidebar to begin.")
        return

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and show_metrics and "metrics" in msg:
                render_metrics(
                    msg["metrics"],
                    layer=msg.get("layer", ""),
                    conf=msg.get("confidence", 0),
                    tokens=msg.get("tokens_used", 0),
                )
            if msg["role"] == "assistant" and show_context and "context" in msg:
                with st.expander("📄 Retrieved Chunks"):
                    for i, chunk in enumerate(msg["context"], 1):
                        clean = re.sub(r"\[Chapter[^\]]*\]\s*", "", chunk).strip()
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
                render_metrics(
                    metrics,
                    layer=result.get("layer", ""),
                    conf=result.get("confidence", 0),
                    tokens=result.get("tokens_used", 0),
                )
            if show_context and result.get("context"):
                with st.expander("📄 Retrieved Chunks"):
                    for i, chunk in enumerate(result["context"], 1):
                        clean = re.sub(r"\[Chapter[^\]]*\]\s*", "", chunk).strip()
                        st.markdown(f"**[Chunk {i}]**\n\n{clean[:400]}{'...' if len(clean)>400 else ''}")

        msg_data = {
            "role":       "assistant",
            "content":    answer,
            "context":    result.get("context", []),
            "layer":      result.get("layer", ""),
            "confidence": result.get("confidence", 0),
            "tokens_used": result.get("tokens_used", 0),
        }
        if metrics:
            msg_data["metrics"] = metrics
        st.session_state.messages.append(msg_data)

if __name__ == "__main__":
    main()
