import os
import re
import json
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. PIB Target URL
PIB_URL = "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2269699&reg=48&lang=2"
OUTPUT_CHUNK_FILE = "data/chunks.json"

def fetch_pib_document(url: str) -> str:
    """Fetches the PIB HTML page and extracts clean text."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    print(f"[1/3] Fetching document from PIB URL...")
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching URL: {e}")
        # Fallback if page download fails: read from a local file if available
        if os.path.exists("data/pib_doc.html"):
            with open("data/pib_doc.html", "r", encoding="utf-8") as f:
                return f.read()
        raise e

    soup = BeautifulSoup(response.content, "html.parser")

    # Remove script, style, header, footer elements
    for element in soup(["script", "style", "header", "footer", "nav", "aside"]):
        element.extract()

    # Locate main content container in PIB pages (usually inside div with class/id)
    content_div = soup.find("div", {"class": "release-content"}) or soup.find("div", {"id": "PrintContent"})
    
    if content_div:
        text = content_div.get_text(separator="\n")
    else:
        text = soup.get_text(separator="\n")

    # Clean excessive whitespaces and newlines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned_text = "\n".join(lines)
    
    print(f"Successfully extracted {len(cleaned_text.split())} words from document.")
    return cleaned_text


def chunk_text(text: str, chunk_size_words: int = 350, overlap_words: int = 50) -> list[dict]:
    """
    Chunks text into ~200-500 word sections.
    1 word ≈ 5-6 characters, so 350 words ≈ 2000 characters.
    """
    print(f"[2/3] Chunking text into ~{chunk_size_words} word sections...")
    
    # Character equivalents for LangChain text splitter
    chunk_size_chars = chunk_size_words * 6  # ~2100 chars
    chunk_overlap_chars = overlap_words * 6  # ~300 chars

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_chars,
        chunk_overlap=chunk_overlap_chars,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

    raw_chunks = splitter.split_text(text)
    
    formatted_chunks = []
    for idx, chunk_text_content in enumerate(raw_chunks):
        word_count = len(chunk_text_content.split())
        
        # Extract a short title or summary topic from the first line
        first_line = chunk_text_content.split("\n")[0][:60]
        
        formatted_chunks.append({
            "chunk_id": f"chunk_{idx + 1:02d}",
            "title": first_line,
            "word_count": word_count,
            "content": chunk_text_content.strip()
        })

    print(f"Generated {len(formatted_chunks)} total chunks.")
    return formatted_chunks


def run_ingestion():
    os.makedirs("data", exist_ok=True)
    
    # Step A: Fetch & Clean Text
    raw_text = fetch_pib_document(PIB_URL)
    
    # Save raw clean text for sanity checks
    with open("data/cleaned_pib_text.txt", "w", encoding="utf-8") as f:
        f.write(raw_text)

    # Step B: Chunk Text
    chunks = chunk_text(raw_text, chunk_size_words=350, overlap_words=50)

    # Step C: Save to data/chunks.json
    print(f"[3/3] Saving chunks to {OUTPUT_CHUNK_FILE}...")
    with open(OUTPUT_CHUNK_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print("\n✅ Step 1 Completed Successfully!")
    
    # Print sample chunk summary
    for c in chunks[:3]:
        print(f" - [{c['chunk_id']}] Words: {c['word_count']} | Title Snippet: {c['title']}...")

if __name__ == "__main__":
    run_ingestion()