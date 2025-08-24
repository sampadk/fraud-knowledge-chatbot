# streamlit_app.py (V6 - Final Version with Synonym Expansion)

import streamlit as st
import re
from langchain_ollama import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# App title and description
st.title("AI Fraud Detection Assistant 🤖")
st.write("This chatbot is powered by a curated knowledge base on fraud typologies and patterns. Ask a question to get started!")

# ==============================================================================
# NEW SECTION: Synonym Map and Expansion Logic
# ==============================================================================

# Your comprehensive synonym map
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
    """
    Expands a user's query with synonyms for better retrieval.
    """
    expanded_query = query
    # Use word boundaries to match whole words/phrases
    for formal_term, aliases in synonym_map.items():
        all_terms = [formal_term] + aliases
        for term in all_terms:
            # Check if the term is in the query (case-insensitive)
            if re.search(r'\b' + re.escape(term) + r'\b', query, re.IGNORECASE):
                # If a match is found, add the formal term and a key alias for context
                # and then break to avoid adding multiple synonyms for the same concept.
                expansion = f" (related to: {formal_term}, {aliases[0]})"
                if expansion not in expanded_query: # Avoid duplicate expansions
                    expanded_query += expansion
                break # Move to the next formal term
    return expanded_query

# ==============================================================================
# The Core RAG Logic (Cached for performance)
# ==============================================================================
@st.cache_resource
def initialize_rag_chain():
    # Load Documents, Split, etc. (same as before)
    loader = DirectoryLoader('knowledge_base/', glob="**/*.txt", show_progress=True)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings)
    llm = OllamaLLM(model="gemma3n:e4b")

    # Your new, more detailed domain glossary
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
    You are an expert fraud detection assistant. Your ONLY task is to answer the user's question based STRICTLY on the context provided below.
    First, use the DOMAIN GLOSSARY to understand key terms. Then, carefully analyze the CONTEXT to find the answer.
    Synthesize the information from the context into a clear, concise, and professional answer.
    Do not use any of your prior knowledge. If the context does not contain the answer, you MUST say: "Based on the provided documents, I do not have enough information to answer that question."

    DOMAIN GLOSSARY:
    {domain_glossary}

    CONTEXT:
    {{context}}

    QUESTION:
    {{question}}

    ANSWER:
    """
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10})

    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=retriever, chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain

# Initialize the chain
qa_chain = initialize_rag_chain()

# ==============================================================================
# Streamlit Chat UI with Query Expansion
# ==============================================================================
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
            # --- NEW: Expand the query before invoking the chain ---
            expanded_prompt = expand_query_with_synonyms(prompt, SYNONYM_MAP)
            
            # Optional: Display the expanded query for debugging/transparency
            if expanded_prompt != prompt:
                st.info(f"Searching with expanded query: *{expanded_prompt}*")

            result = qa_chain.invoke({"query": expanded_prompt})
            response = result["result"]
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})