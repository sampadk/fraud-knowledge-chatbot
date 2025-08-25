# streamlit_app.py (V11 - Cloud-friendly RAG for Fraud Typology Assistant)
# ---------------------------------------------------------------
# Highlights:
# - Removes heavy/unnecessary deps (e.g., unstructured local-inference)
# - Forces lightweight TextLoader for .txt files
# - Stable langchain + Chroma usage with persistence
# - Adds synonym-based query expansion (tiny, fast)
# - Adds glossary-anchored prompt + MMR retrieval
# - Shows citations (file + section) for trust
# - Sidebar controls to rebuild index and tweak k
# ---------------------------------------------------------------
# --- SQLite shim: must be first, before anything imports sqlite3 ---
import sys
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
    sys.modules["pysqlite3"] = pysqlite3
except Exception:
    pass


import os
import re
import time
import traceback
import sqlite3
import streamlit as st
st.caption(f"SQLite version: {sqlite3.sqlite_version}")

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- Environment & app config ---
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CHROMADB_DISABLE_TELEMETRY", "1")

st.set_page_config(page_title="AI Fraud Detection Assistant", page_icon="🛡️")
st.title("AI Fraud Detection Assistant 🤖")
st.caption("Gemini + RAG over your fraud typologies and patterns (markdown .txt files).")

# --- Synonym Map (tiny, high-impact) ---
SYNONYM_MAP = {
    "Authorized Push Payment (APP) Scam": ["safe account scam", "bank transfer scam", "authorized payment scam", "APP", "push payment scam"],
    "Account Takeover (ATO)": ["hijacked account", "compromised account", "login takeover", "session takeover"],
    "Business Email Compromise (BEC)": ["invoice fraud email", "supplier email hack", "CEO fraud", "fake invoice email"],
    "Confirmation of Payee (CoP)": ["name match check", "payee name verification", "beneficiary name match"],
    "Money Mule Activity": ["mule account", "money mule", "pass-through account"],
    "SIM Swap / Number Port-Out": ["sim swap", "number hijack", "port-out", "otp intercept"],
    "Card-Not-Present (CNP) Fraud": ["ecommerce card fraud", "online card fraud"],
    "Card Testing": ["micro authorizations", "card checker", "BIN hammering"],
    "Chargeback / Friendly Fraud": ["refund abuse", "friendly chargeback"],
    "Structuring / Smurfing": ["threshold avoidance", "cash structuring"],
    "Trade-Based Money Laundering (TBML)": ["over/under invoicing", "phantom shipment"],
    "Romance Scam": ["pig-butchering", "pig butchering", "online dating scam"],
    "Phishing / Vishing / Smishing": ["otp phishing", "bank impersonation", "smishing text"],
    "Invoice Fraud / Supplier Impersonation": ["bank detail change scam", "supplier bank switch"],
    "Loyalty / Points Fraud": ["points drain", "rewards theft"],
    "Mobile Malware / RAT": ["remote access tool", "screen overlay", "keylogger"],
    "Credit Bust-Out": ["bustout", "limit max-out then default"],
    "QR Code / Invoice Swap Scam": ["qr swap", "invoice qr replace"]
}

def expand_query_with_synonyms(query: str, synonym_map: dict) -> str:
    """
    Lightweight expansion: when an alias or the formal term appears in the query,
    append a small, single expansion referencing the canonical term to help retrieval.
    """
    expanded_query = query
    for formal_term, aliases in synonym_map.items():
        all_terms = [formal_term] + aliases
        if any(re.search(r"\b" + re.escape(term) + r"\b", query, re.IGNORECASE) for term in all_terms):
            hint = f" (related to: {formal_term})"
            if hint not in expanded_query:
                expanded_query += hint
    return expanded_query

# --- Domain glossary kept tiny so it can be always-on in prompt ---
DOMAIN_GLOSSARY = """
- RAG: Retrieval-Augmented Generation; LLM answers grounded in retrieved documents.
- Typology: Fraud category (the “scheme”) e.g., ATO, APP scam, BEC, mule.
- Pattern: A recurring set of signals that reveal a typology (the “how it shows up”).
- KYC/KYB/CDD/EDD: Identity/business verification; deeper checks for higher risk.
- AML/CTF: Anti–Money Laundering / Counter–Terrorist Financing controls.
- PII/PCI: Personal/Card data that must be protected.
- ATO: Account Takeover via stolen creds, SIM-swap, malware, or social engineering.
- APP Scam: Victim is tricked into authorizing a payment (“safe account”, urgency).
- BEC: Business Email Compromise; spoofed/compromised email drives rogue payments.
- CoP: Confirmation of Payee name-check to reduce misdirected/scam payments.
- CNP/CP: Card-Not-Present (online) / Card-Present (in-person) transaction modes.
- AVS/CVV/3DS/SCA: Card verification + strong customer authentication measures.
- Mule Account: Moves illicit funds (fan-in, short dwell, fan-out).
- SIM Swap/Port-Out: Phone number hijack to intercept OTPs and reset access.
- TBML: Trade-Based Money Laundering via mispriced or fake trade flows.
- UBO: Ultimate Beneficial Owner; real person controlling an entity.
""".strip()

# --- Prompt with strict grounding & a predictable answer frame ---
QA_TEMPLATE = """
You are an expert fraud detection assistant. Answer STRICTLY from the CONTEXT.
If the answer is not in the context, say: "Based on the provided documents, I do not have enough information to answer that question."

DOMAIN GLOSSARY (for term clarity; do not invent definitions):
{domain_glossary}

CONTEXT:
{context}

QUESTION:
{question}

FORMAT:
- Direct answer first (2–5 sentences).
- If relevant, add short bullet points: Typologies, Pattern Name, Key Signals, Prevention.
- End with: "Sources:" followed by the file names/sections used.

ANSWER:
""".strip()

QA_PROMPT = PromptTemplate.from_template(QA_TEMPLATE)

# --- Helpers: load & chunk documents, enrich metadata with section names ---
def _extract_top_heading(text: str) -> str:
    # Grab first markdown heading line (## or #)
    for line in text.splitlines():
        if line.strip().startswith("## "):
            return line.strip()[3:].strip()
        if line.strip().startswith("# "):
            return line.strip()[2:].strip()
    return ""

@st.cache_resource(show_spinner=True)
def build_vectorstore(knowledge_dir: str, persist_dir: str, api_key: str):
    loader = DirectoryLoader(
        knowledge_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,   # force lightweight loader
        show_progress=True
    )
    docs = loader.load()

    # Enrich metadata with a best-effort section title for each chunk later
    splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    for d in chunks:
        # Keep original source path
        source = d.metadata.get("source", "unknown")
        # Guess a section heading for better citations
        section = _extract_top_heading(d.page_content) or "General"
        d.metadata["source"] = source
        d.metadata["section"] = section

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

    vs = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        collection_name="fraud_knowledge",
        persist_directory=persist_dir
    )
    return vs

@st.cache_resource(show_spinner=False)
def init_chain(api_key: str, retriever_k: int = 5, persist_dir: str = ".chroma"):
    # Build or load vectorstore
    vs = build_vectorstore("knowledge_base", persist_dir, api_key)

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.2
        # You can set max_output_tokens here if desired
    )

    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": retriever_k}
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_PROMPT.partial(domain_glossary=DOMAIN_GLOSSARY)},
        return_source_documents=True
    )
    return qa, vs

# --- Sidebar controls ---
with st.sidebar:
    st.header("⚙️ Settings")
    retriever_k = st.slider("Retriever k", min_value=3, max_value=8, value=5, step=1)
    persist_dir = st.text_input("Chroma persist directory", value=".chroma")
    rebuild = st.button("Rebuild index (force re-embed)")
    st.markdown("---")
    st.markdown("**Secrets**: Set `GEMINI_API_KEY` in Streamlit Cloud → App → Settings → Secrets.")

# --- Initialize chain (and optionally rebuild index) ---
api_key = st.secrets.get("GEMINI_API_KEY", "")
if not api_key:
    st.error("Missing `GEMINI_API_KEY` in secrets. Add it in Streamlit Cloud settings.")
    st.stop()

if rebuild:
    # Clear caches, remove previous index
    try:
        st.warning("Rebuilding vectorstore…")
        # wipe persist dir
        import shutil
        if os.path.isdir(persist_dir):
            shutil.rmtree(persist_dir, ignore_errors=True)
        # clear cached resources and rebuild
        build_vectorstore.clear()  # type: ignore
        init_chain.clear()         # type: ignore
        time.sleep(0.2)
    except Exception:
        st.exception(traceback.format_exc())

qa_chain, vectorstore = init_chain(api_key=api_key, retriever_k=retriever_k, persist_dir=persist_dir)

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- Chat input ---
user_q = st.chat_input("Ask about a fraud pattern or typology…")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing…"):
            expanded_q = expand_query_with_synonyms(user_q, SYNONYM_MAP)
            if expanded_q != user_q:
                st.info(f"Expanded query for retrieval: _{expanded_q}_")

            try:
                result = qa_chain.invoke({"query": expanded_q})
                answer = result.get("result", "")
                sources = result.get("source_documents", []) or []

                # Display main answer
                st.markdown(answer if answer else "_No answer produced._")

                # Display citations with file + section
                if sources:
                    st.markdown("**Sources used:**")
                    for i, doc in enumerate(sources, start=1):
                        src = doc.metadata.get("source", "unknown")
                        section = doc.metadata.get("section", "General")
                        st.markdown(f"- {i}. `{os.path.basename(src)}` — _{section}_")

            except Exception:
                st.error("Something went wrong during retrieval or generation.")
                st.code(traceback.format_exc())

    st.session_state.messages.append({"role": "assistant", "content": answer})
