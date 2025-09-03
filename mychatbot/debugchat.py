import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyD32PmVVXbONKbXhvNCdRTaiay5KKZkcpk" # IMPORTANT: Put your key here

try:
    # Step 1: Load the document
    print("STEP 1: Loading document...")
    loader = TextLoader('./wordpress_data.txt', encoding='utf-8')
    documents = loader.load()
    print("...SUCCESS: Document loaded.\n")

    # Step 2: Split the document into chunks
    print("STEP 2: Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    print(f"...SUCCESS: Document split into {len(docs)} chunks.\n")

    # Step 3: Create the embedding model
    print("STEP 3: Initializing embedding model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print("...SUCCESS: Embedding model initialized.\n")

    # Step 4: Create the Chroma vector store
    print("STEP 4: Creating Chroma vector store from documents...")
    # This is the most likely step to fail silently.
    vector_store = Chroma.from_documents(docs, embeddings)
    print("...SUCCESS: Vector store created.\n")

    print("--- RAG SETUP COMPLETED SUCCESSFULLY ---")

except Exception as e:
    print(f"\n--- AN ERROR OCCURRED ---")
    print(f"Error details: {e}")