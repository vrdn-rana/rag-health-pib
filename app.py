import os
import sys
import streamlit as st

# 1. Fetch API Key from Streamlit Secrets or Environment
api_key = (
    st.secrets.get("GEMINI_API_KEY") 
    or os.getenv("GEMINI_API_KEY") 
    or st.secrets.get("GOOGLE_API_KEY") 
    or os.getenv("GOOGLE_API_KEY")
)

# 2. Inject into os.environ
if api_key and len(api_key.strip()) > 10:
    cleaned_key = api_key.strip().strip('"').strip("'")
    os.environ["GEMINI_API_KEY"] = cleaned_key
    os.environ["GOOGLE_API_KEY"] = cleaned_key
    
    # Optional debug message (displays masked key in sidebar for confirmation)
    st.sidebar.success(f"🔑 Key Loaded: `{cleaned_key[:6]}...{cleaned_key[-4:]}`")
else:
    st.sidebar.error("❌ No valid GEMINI_API_KEY found!")
# Ensure project root directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import backend module AFTER environment variable setup
from src.rag_chain import generate_rag_answer

# Page Configuration
st.set_page_config(
    page_title="India's Health Transformation Q&A Assistant",
    page_icon="🏥",
    layout="wide"
)

# Custom Header Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #1E88E5;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# App Title & Description
st.markdown('<div class="main-header">🏥 India\'s Health Transformation Q&A Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Grounded RAG system answering questions strictly using the Press Information Bureau (PIB) Backgrounder Document.</div>',
    unsafe_allow_html=True
)

# Sidebar Controls & Sample Questions
st.sidebar.header("⚙️ Search Settings")
top_k = st.sidebar.slider("Chunks to Retrieve (top-k)", min_value=1, max_value=5, value=3)

st.sidebar.markdown("---")
st.sidebar.header("💡 Sample Questions")
sample_questions = [
    "What is Ayushman Bharat PM-JAY and its key benefits?",
    "What are Ayushman Arogya Mandirs?",
    "What is the Ayushman Bharat Digital Mission (ABDM)?",
    "What key achievements are mentioned under PM-ABHIM?",
]

selected_sample = st.sidebar.radio("Click to auto-fill a question:", ["None"] + sample_questions)

# Question Input Box
default_query = selected_sample if selected_sample != "None" else ""
user_question = st.text_input("Enter your question about India's Health Transformation:", value=default_query, placeholder="e.g. How many Jan Aushadhi Kendras are functional?")

# Submit Button
submit_button = st.button("🔍 Get Answer", type="primary")

# Execute RAG Pipeline when clicked
if submit_button or (selected_sample != "None" and user_question):
    if not user_question.strip():
        st.warning("Please enter a question or select a sample question from the sidebar.")
    else:
        with st.spinner("Searching document & generating grounded answer..."):
            result = generate_rag_answer(user_question, top_k=top_k)
            
            # Extract plain text string from result["answer"]
            answer_data = result["answer"]
            clean_text = ""

            if isinstance(answer_data, list):
                # Extract text fields from content blocks
                text_parts = [
                    item["text"] for item in answer_data 
                    if isinstance(item, dict) and "text" in item
                ]
                clean_text = "\n".join(text_parts) if text_parts else str(answer_data)
            elif isinstance(answer_data, dict):
                clean_text = answer_data.get("text", str(answer_data))
            else:
                clean_text = str(answer_data)

            # Display Answer strictly as clean Markdown text
            st.markdown("### 💬 Answer")
            st.markdown(clean_text)
            
            # Display Source Chunks
            st.markdown("---")
            st.markdown(f"### 📄 Source Context Chunks Used ({len(result['source_chunks'])})")
            
            for idx, chunk in enumerate(result["source_chunks"], 1):
                with st.expander(f"Chunk #{idx}: {chunk['title']} (ID: {chunk['chunk_id']})"):
                    st.markdown(f"**Relevance Distance:** `{chunk['distance']:.4f}`")
                    st.write(chunk["content"])