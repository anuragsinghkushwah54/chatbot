import traceback # Import the traceback module to get detailed error info

# Flask: Creates the Flask web application.
# jsonify: Converts data to JSON format for responses.
# request: Accesses incoming request data.
# send_file: Sends files to the client.
# send_from_directory: Sends files from a specified directory.
from flask import Flask, jsonify, request, send_file, send_from_directory

# --- LangChain Imports for RAG ---
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. Flask App Initialization ---
app = Flask(__name__)

# --- 2. API Key Configuration ---
# IMPORTANT: Make sure this is your actual, working Google API Key.
API_KEY = "AIzaSyD32PmVVXbONKbXhvNCdRTaiay5KKZkcpk"

# --- 3. RAG Pipeline Setup ---
retrieval_chain = None
try:
    print("Setting up the RAG pipeline...")
    loader = TextLoader('./wordpress_data.txt', encoding='utf-8')
    documents = loader.load()
    
    # Tuned text splitter for better context
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
    vector_store = Chroma.from_documents(docs, embeddings)
    
    # Tuned retriever to fetch more documents
    retriever = vector_store.as_retriever(search_kwargs={"k": 7})
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=API_KEY)
    
    # --- FIXED PROMPT ---
    # This prompt is complete and gives clear instructions to the AI.
    prompt_template = ChatPromptTemplate.from_template("""
    You are an expert assistant for the BioAge website. Your task is to answer user questions accurately based ONLY on the provided context.

    Instructions:
    1. Read the following context carefully.
    2. Synthesize a comprehensive and helpful answer to the user's question using only this context.
    3. If the answer is not found within the context, you MUST respond with: "I'm sorry, I don't have information on that topic based on the provided articles."
    4. Do not use any external knowledge or make up information.

    <context>
    {context}
    </context>

    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    print("RAG pipeline setup complete.")
except Exception as e:
    print(f"FATAL ERROR during RAG setup: {e}")
    traceback.print_exc()

# --- 4. Flask Routes ---

@app.route('/')
def home():
    return send_file('web/main.html')

@app.route("/api/generate", methods=["POST"])
def generate_api():
    print("\n--- Received a request at /api/generate ---")
    if retrieval_chain is None:
        print("Error: RAG pipeline is not initialized.")
        return jsonify({"error": "RAG pipeline is not initialized. Check server logs."}), 500

    try:
        req_body = request.get_json()
        user_input = req_body.get("contents")
        print(f"User input received: '{user_input}'")

        if not user_input:
            print("Error: No input provided by the user.")
            return jsonify({"error": "No input provided"}), 400

        print("Invoking the retrieval chain...")
        response = retrieval_chain.invoke({"input": user_input})
        print("...Retrieval chain finished successfully.")

        final_answer = response.get("answer", "Sorry, something went wrong processing the answer.")
        print(f"Generated answer: '{final_answer}'")
        
        return jsonify({"text": final_answer})

    except Exception as e:
        print(f"\n!!!!!! AN ERROR OCCURRED IN /api/generate !!!!!!")
        print(f"Error details: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred. Check the server logs for details."}), 500

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('web', path)

# --- 5. Main Execution Block ---
if __name__ == '__main__':
    app.run(debug=True)

