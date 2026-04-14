import streamlit as st
import requests
import uuid

# --- Page Config ---
st.set_page_config(
    page_title="DocuMind — Multi-Domain RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    /* Base */
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #0a0a0f;
        color: #e2e8f0;
    }

    .stApp {
        background-color: #0a0a0f;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0f0f1a;
        border-right: 1px solid #1e1e2e;
    }

    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Chat messages */
    .user-message {
        background: #1a1a2e;
        border: 1px solid #2d2d4a;
        border-radius: 12px 12px 4px 12px;
        padding: 14px 18px;
        margin: 8px 0;
        margin-left: 15%;
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.95rem;
        line-height: 1.6;
        color: #e2e8f0;
    }

    .assistant-message {
        background: #0f1117;
        border: 1px solid #1e2a3a;
        border-left: 3px solid #3b82f6;
        border-radius: 4px 12px 12px 12px;
        padding: 14px 18px;
        margin: 8px 0;
        margin-right: 15%;
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.95rem;
        line-height: 1.6;
        color: #cbd5e1;
    }

    .assistant-message.no-confidence {
        border-left: 3px solid #f59e0b;
        background: #0f0f0a;
    }

    /* Source pills */
    .source-pill {
        display: inline-block;
        background: #1e2a3a;
        border: 1px solid #2d3f52;
        color: #64a8d8 !important;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        padding: 2px 10px;
        border-radius: 20px;
        margin: 4px 3px 0 0;
    }

    /* Domain card */
    .domain-card {
        background: #0f1117;
        border: 1px solid #1e2a3a;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }

    .domain-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: #3b82f6;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 4px;
    }

    .domain-desc {
        font-size: 0.82rem;
        color: #94a3b8;
        line-height: 1.5;
    }

    /* Header */
    .app-header {
        padding: 20px 0 10px 0;
        border-bottom: 1px solid #1e1e2e;
        margin-bottom: 20px;
    }

    .app-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.4rem;
        font-weight: 500;
        color: #f1f5f9;
        letter-spacing: -0.02em;
    }

    .app-subtitle {
        font-size: 0.82rem;
        color: #475569;
        margin-top: 4px;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        background: #0f2a1a;
        border: 1px solid #166534;
        color: #4ade80;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        padding: 3px 10px;
        border-radius: 20px;
    }

    /* Input */
    .stTextInput input, .stTextArea textarea {
        background-color: #0f1117 !important;
        border: 1px solid #2d2d4a !important;
        color: #e2e8f0 !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        border-radius: 8px !important;
    }

    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 1px #3b82f6 !important;
    }

    /* Buttons */
    .stButton button {
        background-color: #1e2a3a !important;
        color: #e2e8f0 !important;
        border: 1px solid #2d3f52 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.8rem !important;
        border-radius: 6px !important;
        transition: all 0.2s ease !important;
    }

    .stButton button:hover {
        background-color: #2d3f52 !important;
        border-color: #3b82f6 !important;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background-color: #0f1117 !important;
        border: 1px solid #2d2d4a !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
    }

    /* Divider */
    hr {
        border-color: #1e1e2e !important;
    }

    /* Scrollable chat area */
    .chat-container {
        max-height: 65vh;
        overflow-y: auto;
        padding-right: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- Constants ---
API_URL = "http://localhost:8000"

DOMAIN_METADATA = {
    "zomato": {
        "label": "📊 Zomato Annual Report 2023",
        "desc": "Business performance, financials, GMV, strategy, and market position",
        "tag": "FINANCIAL"
    },
    "rbi": {
        "label": "🏦 RBI Monetary Policy Report",
        "desc": "Inflation outlook, interest rate decisions, GDP growth, economic indicators",
        "tag": "REGULATORY"
    },
    "dpdp": {
        "label": "⚖️ DPDP Act 2023",
        "desc": "India's data privacy law — rights, obligations, penalties, compliance",
        "tag": "LEGAL"
    }
}

# --- Session State Init ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_domain" not in st.session_state:
    st.session_state.selected_domain = "dpdp"

# --- API Helpers ---
def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except:
        return False

def send_message(question: str, domain: str) -> dict:
    try:
        r = requests.post(f"{API_URL}/chat", json={
            "session_id": st.session_state.session_id,
            "domain": domain,
            "question": question
        }, timeout=30)
        return r.json()
    except Exception as e:
        return {"answer": f"API error: {str(e)}", "confident": False, "sources": []}

def reset_session():
    try:
        requests.post(f"{API_URL}/reset", json={
            "session_id": st.session_state.session_id
        }, timeout=5)
    except:
        pass
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())[:8]

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 20px 0;'>
        <div style='font-family: IBM Plex Mono, monospace; font-size: 1.1rem; 
                    font-weight: 500; color: #f1f5f9;'>⚡ DocuMind</div>
        <div style='font-family: IBM Plex Mono, monospace; font-size: 0.65rem; 
                    color: #475569; margin-top: 4px;'>MULTI-DOMAIN RAG</div>
    </div>
    """, unsafe_allow_html=True)

    # API Status
    api_healthy = check_api_health()
    if api_healthy:
        st.markdown('<span class="status-badge">● API ONLINE</span>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<span class="status-badge" style="background:#2a0f0f;border-color:#7f1d1d;'
            'color:#f87171;">● API OFFLINE</span>',
            unsafe_allow_html=True
        )
        st.warning("Start the FastAPI server: `uvicorn main:app --port 8000`")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family: IBM Plex Mono, monospace; font-size: 0.7rem; '
        'color: #475569; text-transform: uppercase; letter-spacing: 0.1em; '
        'margin-bottom: 8px;">Select Domain</div>',
        unsafe_allow_html=True
    )

    # Domain selector
    domain_options = list(DOMAIN_METADATA.keys())
    domain_labels = [DOMAIN_METADATA[d]["label"] for d in domain_options]
    selected_idx = st.selectbox(
        "domain",
        range(len(domain_options)),
        format_func=lambda i: domain_labels[i],
        label_visibility="collapsed"
    )
    selected_domain = domain_options[selected_idx]

    # Domain info card
    meta = DOMAIN_METADATA[selected_domain]
    st.markdown(f"""
    <div class="domain-card">
        <div class="domain-label">{meta['tag']}</div>
        <div class="domain-desc">{meta['desc']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Session info
    st.markdown(
        f'<div style="font-family: IBM Plex Mono, monospace; font-size: 0.7rem; '
        f'color: #475569;">SESSION · {st.session_state.session_id}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div style="font-family: IBM Plex Mono, monospace; font-size: 0.7rem; '
        f'color: #475569; margin-top: 4px;">'
        f'MESSAGES · {len(st.session_state.messages)}</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("↺  New Conversation", use_container_width=True):
        reset_session()
        st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family: IBM Plex Mono, monospace; font-size: 0.65rem; 
                color: #334155; line-height: 1.8;'>
        STACK<br>
        LangChain · ChromaDB<br>
        HuggingFace · Groq<br>
        FastAPI · Streamlit
    </div>
    """, unsafe_allow_html=True)

# --- Main Area ---
col1, col2 = st.columns([6, 1])
with col1:
    st.markdown("""
    <div class="app-header">
        <div class="app-title">Document Intelligence</div>
        <div class="app-subtitle">
            Query financial, regulatory & legal documents · Powered by RAG
        </div>
    </div>
    """, unsafe_allow_html=True)

# Handle domain switch — clear messages if domain changed
if selected_domain != st.session_state.selected_domain:
    st.session_state.selected_domain = selected_domain
    st.session_state.messages = []
    reset_session()

# --- Chat History ---
if not st.session_state.messages:
    domain_meta = DOMAIN_METADATA[selected_domain]
    st.markdown(f"""
    <div style='text-align: center; padding: 60px 20px;'>
        <div style='font-size: 2rem; margin-bottom: 12px;'>
            {domain_meta['label'].split()[0]}
        </div>
        <div style='font-family: IBM Plex Sans, sans-serif; font-size: 1rem; 
                    color: #475569; max-width: 400px; margin: 0 auto; line-height: 1.6;'>
            Ask anything about the<br>
            <span style='color: #94a3b8;'>{domain_meta['label'][2:]}</span>
        </div>
        <div style='margin-top: 24px; font-family: IBM Plex Mono, monospace; 
                    font-size: 0.7rem; color: #334155;'>
            HYBRID SEARCH · MULTI-TURN MEMORY · CONFIDENCE THRESHOLD
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-message">{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            confidence_class = "" if msg.get("confident", True) else " no-confidence"
            st.markdown(
                f'<div class="assistant-message{confidence_class}">'
                f'{msg["content"]}</div>',
                unsafe_allow_html=True
            )
            if msg.get("sources"):
                sources_html = "".join([
                    f'<span class="source-pill">pg {s}</span>'
                    for s in msg["sources"]
                ])
                st.markdown(
                    f'<div style="margin: -4px 0 12px 0;">{sources_html}</div>',
                    unsafe_allow_html=True
                )

# --- Input ---
st.markdown("<br>", unsafe_allow_html=True)
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([8, 1])
    with col1:
        user_input = st.text_input(
            "question",
            placeholder=f"Ask a question about the {DOMAIN_METADATA[selected_domain]['label'][2:]}...",
            label_visibility="collapsed"
        )
    with col2:
        submitted = st.form_submit_button("Send", use_container_width=True)

if submitted and user_input.strip() and api_healthy:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Get response
    with st.spinner(""):
        response = send_message(user_input, selected_domain)

    # Add assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "confident": response["confident"],
        "sources": response.get("sources", [])
    })

    st.rerun()