# streamlit_app.py (V14 – Flash-only, with helpful sidebar hints)

import os
import re
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Optional compression (auto-disabled if not available in your LangChain build)
try:
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain_community.document_transformers import EmbeddingsRedundantFilter
    HAS_COMPRESSION = True
except Exception:
    HAS_COMPRESSION = False

st.title("AI Fraud Risk Whisperer 🤖")
st.write("Gemini 1.5 Flash + RAG over your fraud knowledge base. Ask away!")

# ---------- Synonym Map ----------
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
    expanded_query = query
    for formal_term, aliases in synonym_map.items():
        all_terms = [formal_term] + aliases
        for term in all_terms:
            if re.search(r"\b" + re.escape(term) + r"\b", query, re.IGNORECASE):
                hint = f" (related to: {formal_term})"
                if hint not in expanded_query:
                    expanded_query += hint
                break
    return expanded_query

# ---------- Sidebar controls (with descriptions) ----------
with st.sidebar:
    st.header("⚙️ Settings")
    temperature = st.slider(
        "Temperature",
        0.0, 1.0, 0.25, 0.05,
        help="Controls creativity. Lower = more factual/concise; higher = more varied wording."
    )
    retriever_k = st.slider(
        "Retriever k (MMR)",
        3, 8, 6, 1,
        help="How many chunks to fetch before answering. Higher can recall more, but may add noise."
    )
    compress_requested = st.checkbox(
        "Contextual compression",
        value=True if HAS_COMPRESSION else False,
        help="Removes redundant sentences in retrieved chunks to keep context tight. (Auto-disables if unsupported.)"
    )
    if compress_requested and not HAS_COMPRESSION:
        st.caption("Compression not available in this LangChain build; using base retriever.")
    st.markdown("---")
    st.caption("Set `GEMINI_API_KEY` in your Streamlit app secrets.")

# ---------- Glossary, few-shots, and prompt ----------
DOMAIN_GLOSSARY = """
- RAG: Retrieval-Augmented Generation; LLM answers grounded in retrieved documents.
- Typology: Fraud category (the “scheme”) e.g., ATO, APP scam, BEC, mule.
- Pattern: Recurring signals that reveal a typology (the “how it shows up”).
- KYC/KYB/CDD/EDD: Identity/business verification; deeper checks for higher risk.
- AML/CTF: Anti–Money Laundering / Counter–Terrorist Financing controls.
- PII/PCI: Personal/Card data requiring protection.
- ATO: Account Takeover via stolen creds, SIM-swap, malware, or social engineering.
- APP Scam: Victim is tricked into authorizing a payment (“safe account”, urgency).
- BEC: Business Email Compromise; spoofed/compromised email drives rogue payments.
- CoP: Confirmation of Payee name-check to reduce misdirected/scam payments.
- AVS/CVV/3DS/SCA: Card verification + strong customer authentication measures.
- Mule Account: Moves illicit funds (fan-in, short dwell, fan-out).
- TBML: Trade-Based Money Laundering via mispriced/fake trade flows.
""".strip()

FEW_SHOTS = """
EXAMPLES
User: "hey"
Frawis: "Hi there — happy to help. Want to explore a fraud typology, a detection pattern, or a quick prevention checklist?"

User: "tell me about safe accounts"
Frawis: "In fraud contexts, “safe account” usually refers to an APP (Authorized Push Payment) scam narrative. I can outline common signals and prevention. Which scenario fits: a brand-new payee with urgency, or a customer who ignored an in-app warning?"

User: "what’s the capital of France?"
Frawis: "I’m Frawis, focused on fraud typologies, patterns, and prevention. If helpful, I can show how social engineering pushes victims into off-platform payments."
""".strip()

SYSTEM_PROMPT = f"""
You are **Frawis — the AI Fraud Risk Whisperer**.
Audience: fraud analysts, PMs, and learners. Be clear, calm, and helpful.

CORE RULES
1) **Grounding first**: Prefer the supplied CONTEXT snippets. If answers require synthesis, combine snippets and avoid repeating the same sentence twice.
2) **No guessing**: If the context isn’t enough, say so briefly, offer the 1–2 most relevant follow-up questions or related topics you *can* answer from context, then stop.
3) **Tone**: Professional but human. One friendly sentence is welcome (“Happy to help.”) but avoid jokes unless the user jokes first.
4) **Structure**:
   - Start with a 2–4 sentence direct answer.
   - If useful, add short bullets with: *Typologies*, *Pattern name*, *Key signals*, *Prevention*, *Next steps*.
   - End with **Sources:** and list the file/section names you used (no links).
5) **Citations discipline**: Only cite items in the provided CONTEXT. Never cite memory or outside knowledge.

OFF-TOPIC & SMALL TALK
- Greetings/thanks: respond warmly in 1–2 sentences and ask a helpful follow-up.
- Non-fraud: reply once with a polite redirection. If they persist, keep it minimal (≤3 sentences), clearly mark it as general knowledge, and do **not** use citations.

SAFETY & CLARITY
- Do not invent regulations, vendors, or statistics not in CONTEXT.
- Expand acronyms once (e.g., “APP—Authorized Push Payment scam”).
- If lists are long, group into 3–5 bullets with the highest-signal items first.

{FEW_SHOTS}
"""

QA_TEMPLATE = """
{system}

<GLOSSARY>
{glossary}
</GLOSSARY>

<QUESTION>
{query}
</QUESTION>

<CONTEXT>
{context}
</CONTEXT>

<OUTPUT_REQUIREMENTS>
- Answer only from CONTEXT; if insufficient, say so briefly and suggest 1–2 precise follow-ups.
- Keep the opening answer to 2–4 sentences; then bullets if helpful.
- Normalize ambiguous user terms once (e.g., “safe account” → APP scam).
- Close with “Sources:” followed by file and section names from CONTEXT.
</OUTPUT_REQUIREMENTS>

ANSWER:
""".strip()

QA_CHAIN_PROMPT = PromptTemplate.from_template(QA_TEMPLATE)

def _extract_top_heading(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("## "):
            return s[3:].strip()
        if s.startswith("# "):
            return s[2:].strip()
    return ""

# ---------- Build the RAG chain (Flash-only) ----------
@st.cache_resource(show_spinner=True)
def initialize_rag_chain(temp: float, retriever_k: int, use_compression: bool):
    loader = DirectoryLoader(
        "knowledge_base/",
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True
    )
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=380, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    for d in chunks:
        d.metadata["section"] = _extract_top_heading(d.page_content) or "Section"

    # Prefer newer embed model if available; fallback gracefully
    emb_model = os.environ.get("GEMINI_EMBED_MODEL", "models/text-embedding-004")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=emb_model, google_api_key=st.secrets["GEMINI_API_KEY"])
    except Exception:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=st.secrets["GEMINI_API_KEY"])

    vectorstore = FAISS.from_documents(chunks, embeddings)

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        google_api_key=st.secrets["GEMINI_API_KEY"],
        temperature=temp,
        top_p=0.9,
        max_output_tokens=768,
    )

    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": retriever_k, "lambda_mult": 0.5}
    )

    if use_compression and HAS_COMPRESSION:
        compressor = EmbeddingsRedundantFilter(embeddings=embeddings, similarity_threshold=0.95)
        retriever = ContextualCompressionRetriever(
            base_retriever=base_retriever,
            base_compressor=compressor
        )
    else:
        retriever = base_retriever

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": QA_CHAIN_PROMPT.partial(system=SYSTEM_PROMPT, glossary=DOMAIN_GLOSSARY)
        },
        return_source_documents=True
    )
    return qa_chain

qa_chain = initialize_rag_chain(
    temp=temperature,
    retriever_k=retriever_k,
    use_compression=compress_requested
)

# ---------- Chat UI ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask about a fraud typology, pattern, or signals…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing…"):
            expanded = expand_query_with_synonyms(prompt, SYNONYM_MAP)
            if expanded != prompt:
                st.info(f"Expanded query for retrieval: _{expanded}_")

            result = qa_chain.invoke({"query": expanded})
            answer = result.get("result", "")
            sources = result.get("source_documents", []) or []

            st.markdown(answer if answer else "_No answer produced._")
            if sources:
                st.markdown("**Sources used:**")
                for i, doc in enumerate(sources, start=1):
                    fname = os.path.basename(doc.metadata.get("source", "document.txt"))
                    section = doc.metadata.get("section", "Section")
                    st.markdown(f"- {i}. `{fname}` — _{section}_")

    st.session_state.messages.append({"role": "assistant", "content": answer})