# --- Core Imports for DB Setup ---
import os
import time
import traceback
import sys

# Assume the module is correctly named 'fetch.py'
try:
    import fetch
except ImportError:
    print("FATAL: Cannot import fetch.py. Ensure the file exists.")
    sys.exit(1)

# --- Qdrant and LangChain Imports ---
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse 
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

# --- 1. Centralized Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "zalgo-rag-collection"
EMBEDDING_DIMENSION = 1536 

if not OPENAI_API_KEY or not QDRANT_HOST or not QDRANT_API_KEY:
    print("FATAL: Missing OPENAI_API_KEY, QDRANT_HOST, or QDRANT_API_KEY.")
    sys.exit(1)


def create_and_populate_qdrant():
    """
    Handles data fetching, chunking, embedding, and uploading to Qdrant.
    This function should only be run ONCE locally.
    """
    print("--- Starting Qdrant Database Population ---")

    # 1. Initialize Clients (Timeout is set high for slow network uploads)
    qdrant_client = QdrantClient(
        url=QDRANT_HOST, 
        api_key=QDRANT_API_KEY,
        timeout=300 # 5 minutes max timeout
    )
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    # 2. Wait for data file 
    print("Waiting for data file...")
    max_wait_time = 60
    wait_start = time.time()
    
    # Check if the output file exists using the reference from fetch.py
    if not hasattr(fetch, 'OUTPUT_FILE'):
        print("FATAL: fetch module is missing 'OUTPUT_FILE' attribute.")
        sys.exit(1)
        
    while not os.path.exists(fetch.OUTPUT_FILE):
        if time.time() - wait_start > max_wait_time:
            raise FileNotFoundError("Initial data file missing.")
        time.sleep(2)
    print("Data file found.")

    # 3. Load and Chunk Data
    loader = TextLoader(fetch.OUTPUT_FILE, encoding='utf-8')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100) 
    docs = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} documents, split into {len(docs)} chunks.")

    # 4. Recreate Qdrant Collection 
    try:
        qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"Deleted old collection: {COLLECTION_NAME}")
    except UnexpectedResponse:
        pass 

    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIMENSION,
            distance=models.Distance.COSINE
        )
    )
    print("Collection recreated on remote Qdrant.")

    # 5. Insert Documents (The heavy lifting)
    print("Starting document embedding and upserting (this will take time)...")
    
    # Use batch_size=10, which you confirmed was stable
    QdrantVectorStore.from_documents(
        docs, 
        embeddings, 
        url=QDRANT_HOST,       
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        batch_size=10 # Confirmed stable batch size
    )
    
    print("✅ RAG Database Population Complete!")


if __name__ == "__main__":
    try:
        create_and_populate_qdrant()
    except Exception as e:
        print(f"FATAL ERROR during DB build: {e}")
        traceback.print_exc()
        sys.exit(1)
