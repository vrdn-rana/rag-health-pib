import os
import json
import chromadb
from chromadb.utils import embedding_functions

# Paths and Constants
CHUNKS_FILE = "data/chunks.json"
CHROMA_DB_DIR = "data/chroma_db"
COLLECTION_NAME = "pib_health_chunks"

# 1. Initialize Open-Source Embedding Function (Local & Free)
# Using all-MiniLM-L6-v2 model from Hugging Face
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def build_vector_database():
    """Reads chunks from data/chunks.json and populates ChromaDB."""
    if not os.path.exists(CHUNKS_FILE):
        raise FileNotFoundError(f"Missing {CHUNKS_FILE}. Please run Step 1 (src/ingestion.py) first!")

    # Read chunks generated in Step 1
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"[1/2] Connecting to ChromaDB at '{CHROMA_DB_DIR}'...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # Recreate collection to ensure clean state
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}  # Cosine similarity metric
    )

    print(f"[2/2] Generating embeddings and inserting {len(chunks)} chunks...")
    
    ids = []
    documents = []
    metadatas = []

    for item in chunks:
        ids.append(item["chunk_id"])
        documents.append(item["content"])
        metadatas.append({
            "title": item["title"],
            "word_count": item["word_count"]
        })

    # Add to ChromaDB vector store
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    print(f"\n✅ Step 2 Completed Successfully! ChromaDB store saved at '{CHROMA_DB_DIR}'.")


def search_similar_chunks(query: str, top_k: int = 3):
    """
    Given a user query, computes query embedding and retrieves top_k similar chunks.
    """
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = chroma_client.get_collection(
        name=COLLECTION_NAME, 
        embedding_function=embedding_fn
    )

    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    retrieved_chunks = []
    for idx in range(len(results["ids"][0])):
        retrieved_chunks.append({
            "chunk_id": results["ids"][0][idx],
            "title": results["metadatas"][0][idx]["title"],
            "distance": results["distances"][0][idx],  # Cosine distance
            "content": results["documents"][0][idx]
        })

    return retrieved_chunks


if __name__ == "__main__":
    # 1. Build the database
    build_vector_database()

    # 2. Test semantic search with a sample question
    print("\n--- Testing Semantic Search ---")
    test_query = "What is Ayushman Bharat PM-JAY and its benefits?"
    print(f"Query: '{test_query}'\n")

    top_chunks = search_similar_chunks(test_query, top_k=2)
    for i, chunk in enumerate(top_chunks, 1):
        print(f"Match #{i} [{chunk['chunk_id']}] (Distance: {chunk['distance']:.4f})")
        print(f"Title: {chunk['title']}")
        print(f"Content Snippet: {chunk['content'][:200]}...\n")