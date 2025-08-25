# streamlit_app.py (V12 - Stable Dependencies Version)

import streamlit as st
import re
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader # Ensure this is here
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# App title and description
st.title("AI Fraud Risk Whisperer 🤖")
st.write("This chatbot is powered by Google's Gemini model and a curated knowledge base (using RAG). Ask a question to get started!")

# Your full Synonym Map
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

def expand_query_with_synonyms(query, synonym_map):
    expanded_query = query
    for formal_term, aliases in synonym_map.items():
        all_terms = [formal_term] + aliases
        for term in all_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', query, re.IGNORECASE):
                expansion = f" (related to: {formal_term}, {aliases[0]})"
                if expansion not in expanded_query:
                    expanded_query += expansion
                break
    return expanded_query

@st.cache_resource
def initialize_rag_chain():
    # Load Documents using the specified TextLoader for .txt files
    loader = DirectoryLoader(
        'knowledge_base/',
        glob="**/*.txt",
        loader_cls=TextLoader, # <-- THE CRITICAL CHANGE
        show_progress=True
    )
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=st.secrets["GEMINI_API_KEY"]
    )
    
    #vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings)
    vectorstore = FAISS.from_documents(documents=texts, embedding=embeddings)
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=st.secrets["GEMINI_API_KEY"], 
        temperature=0.3
    )
    
    domain_glossary = """
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
    """

    # Create the RAG Chain with the Final, Supercharged Prompt
    template = f"""
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
        - Greetings/thanks: respond warmly in 1–2 sentences and ask a helpful follow-up (“Want to check a typology or a pattern?”).
        - Non-fraud questions: reply with a polite redirection in one sentence: “I’m Frawis, focused on fraud typologies, patterns, and prevention. What would you like to explore?”
        - If the user still insists on non-fraud content, keep it minimal (≤3 sentences), clearly mark it as general knowledge, and do **not** use citations.

    SAFETY & CLARITY
        - Do not invent regulations, case law, vendor names, or statistics not in CONTEXT.
        - Prefer clear definitions over jargon. Expand acronyms once (e.g., “APP—Authorized Push Payment scam”).
        - If lists are long, group them into 3–5 bullets with the highest signal first.

    DOMAIN GLOSSARY:
    {domain_glossary}

    CONTEXT:
    {{context}}

    QUESTION:
    {{question}}

    <OUTPUT_REQUIREMENTS>
        - Answer only from CONTEXT; if insufficient, say so briefly and suggest 1–2 precise follow-ups.
        - Keep the opening answer to 2–4 sentences; then bullets if helpful.
        - Use the user’s terms; if they’re ambiguous (e.g., “safe account”), normalize once (APP scam).
        - Close with “Sources:” followed by file and section names from CONTEXT.
    </OUTPUT_REQUIREMENTS>

    ANSWER:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 8})
    
    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=retriever, chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain

# Initialize the chain
qa_chain = initialize_rag_chain()

# Streamlit Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about a fraud pattern or typology..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            expanded_prompt = expand_query_with_synonyms(prompt, SYNONYM_MAP)
            
            if expanded_prompt != prompt:
                st.info(f"Searching with expanded query: *{expanded_prompt}*")

            result = qa_chain.invoke({"query": expanded_prompt})
            response = result["result"]
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})