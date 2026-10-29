import os
import sys

# Ensure project root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.vector_store import search_similar_chunks

# Load local environment if running locally
load_dotenv()


def get_active_api_key() -> str:
    """Dynamically retrieve clean API key from environment."""
    raw_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    return raw_key.strip().strip('"').strip("'")


def format_context(retrieved_chunks: list[dict]) -> str:
    """Formats retrieved vector store chunks into string context."""
    context_str = ""
    for idx, chunk in enumerate(retrieved_chunks, 1):
        context_str += f"--- CONTEXT CHUNK {idx} (Title: {chunk['title']}) ---\n"
        context_str += f"{chunk['content']}\n\n"
    return context_str


def generate_rag_answer(user_question: str, top_k: int = 3) -> dict:
    api_key = get_active_api_key()

    if not api_key:
        return {
            "question": user_question,
            "answer": "Error: GEMINI_API_KEY is missing or empty. Please set GEMINI_API_KEY in Streamlit Secrets.",
            "source_chunks": []
        }

    # 1. Retrieve relevant chunks from ChromaDB
    chunks = search_similar_chunks(user_question, top_k=top_k)
    formatted_context = format_context(chunks)

    # 2. Strict Prompt Template forcing clean plain text
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a precise AI assistant answering questions about India's Health Transformation.

CRITICAL FORMATTING INSTRUCTIONS:
- Respond in clean, beautifully structured Markdown text.
- Do NOT output JSON, dictionary lists, or metadata.
- Answer strictly based on the provided PIB context below.

--- PROVIDED CONTEXT ---
{context}

--- USER QUESTION ---
{question}

--- GROUNDED ANSWER ---"""
    )

    candidate_models = [
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite"
    ]

    clean_answer = None
    last_error = None
    used_model = None

    for model_name in candidate_models:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.2,
                max_retries=3
            )

            chain = prompt_template | llm
            response = chain.invoke({
                "context": formatted_context,
                "question": user_question
            })

            # Extract pure string content
            if hasattr(response, "content"):
                res_content = response.content
                # If res_content is a list (e.g. multi-modal block), parse out the text block
                if isinstance(res_content, list):
                    text_parts = [item.get("text", "") for item in res_content if isinstance(item, dict) and "text" in item]
                    clean_answer = "\n".join(text_parts) if text_parts else str(res_content)
                else:
                    clean_answer = str(res_content)

            if clean_answer and len(clean_answer.strip()) > 0:
                used_model = model_name
                break
        except Exception as e:
            last_error = str(e)
            continue

    if not clean_answer:
        clean_answer = f"Error generating answer: {last_error}"

    return {
        "question": user_question,
        "answer": clean_answer.strip(),
        "source_chunks": chunks,
        "model_used": used_model
    }