# --- Core Imports ---
import traceback
import threading
import os
import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from operator import itemgetter 

# --- Google Sheets Imports ---
import gspread
from google.oauth2.service_account import Credentials

# --- LangChain/Qdrant Imports for RAG (FAST COLD START) ---
from qdrant_client import QdrantClient # Needed to initialize the client
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_qdrant import QdrantVectorStore # Used for remote connection

# --- External Modules ---
# NOTE: fetch is no longer used for data loading/threading in app.py
# import fetch # REMOVED: No longer needed for background fetch or setup

# --- 1. Centralized Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAPS_API_KEY = os.getenv("MAPS_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST") # NEW: Required for remote Qdrant
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") # NEW: Required for Qdrant Auth

COLLECTION_NAME = "zalgo-rag-collection"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
# IMPORTANT: This must be in your project root on Vercel
SERVICE_ACCOUNT_FILE = 'credentials.json' 
SHEET_NAME = "Zalgochatbot"

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not found.")


# --- 2. Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Reverse Geocoding Function (Kept) ---
def reverse_geocode(lat, lon):
    """Converts latitude and longitude to a detailed address."""
    API_KEY = MAPS_API_KEY 
    if not API_KEY:
        return f"Lat: {lat}, Lon: {lon} (Geocoding API Key Missing)"
        
    base_url = "https://geocode.maps.co/reverse"
    params = {"lat": lat, "lon": lon, "api_key": API_KEY}
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get('display_name', f"Lat: {lat}, Lon: {lon}")
    except requests.exceptions.RequestException:
        return f"Lat: {lat}, Lon: {lon} (Geocode Failed)"

# --- 3. Google Sheets Logging Function (Kept) ---
def log_to_google_sheet(user_data):
    """Appends a row of data to the specified Google Sheet."""
    try:
        # NOTE: This relies on 'credentials.json' being present
        creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        client = gspread.authorize(creds)
        sheet = client.open(SHEET_NAME).sheet1 
        sheet.append_row(user_data)
    except Exception as e:
        print(f"Google Sheet Logging Error: {e}")
        traceback.print_exc()

# --- 4. Background Data Fetching (REMOVED) ---
# fetch_thread = threading.Thread(target=fetch.run_fetch_loop, daemon=True)
# fetch_thread.start() 
# ^ DELETED: This block causes timeouts on Serverless platforms.

# --- 5. RAG Pipeline Setup (SIMPLIFIED FOR FAST COLD START) ---
retrieval_chain = None
retriever = None 
try:
    print("RAG Setup: Initializing clients and connecting to existing Qdrant collection...")
    
    # 1. Initialize Clients (Fast connections to external services)
    qdrant_client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY) 

    # 2. Connect to Existing Collection (Crucial: NO data loading/upserting here)
    # The data must already be populated by the separate 'build_db.py' script.
    vector_store = QdrantVectorStore(
        client=qdrant_client, 
        embeddings=embeddings, 
        collection_name=COLLECTION_NAME
    )
    
    retriever = vector_store.as_retriever(
        search_kwargs={'k': 6}
    )
    
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini")

    prompt_template = ChatPromptTemplate.from_template("""
    You are an expert assistant for the website. Your task is to answer user questions accurately based ONLY on the provided context.

    Instructions:
    1. Read the following context carefully.
    2. Synthesize a comprehensive and helpful answer. You **MUST** include the most relevant product link (e.g., https://bioage.com/biosuperfood-f2k/) found in the provided context in your final response if question is about products.
    3. If the answer is NOT present in the context, you MUST politely state the information is unavailable, and then provide the contact phone number (877-288-9116) and email (info@bioage.com).
    4. Answer the user's question directly, using the user's language.
    5. Be **concise yet complete**, ensuring all necessary details (like product names and prices) are included if applicable.
    6. maximum answer length must be under 200 characters.
    
    The user's name is {user_name} and their email is {user_email}.

    <context>
    {context}
    </context>

    Question: {input}
    """)
    
    def format_docs(docs):
        """Helper function to format documents into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)

    retrieval_chain = (
        {
            "context": itemgetter("input") | retriever | format_docs, 
            "input": itemgetter("input"),
            "user_name": itemgetter("user_name"),
            "user_email": itemgetter("user_email")
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    print("✅ RAG pipeline setup complete (connected to existing remote DB).")
except Exception as e:
    print(f"FATAL RAG CONNECTION FAILURE (Check Vercel Environment Variables): {e}")
    retriever = None # Explicitly set to None so the API route returns a clean error
    traceback.print_exc()

# --- 6. Flask Route ---
@app.route("/api/generate", methods=["POST"])
def generate_api():
    
    global retriever
    if retriever is None:
        # Returns 500 if the initialization failed in the try block above
        return jsonify({"error": "RAG system failed to initialize. Check Vercel logs for API key or connection errors."}), 500

    try:
        req_body = request.get_json()
        user_input = req_body.get("contents")
        user_name = req_body.get("userName", "User") 
        user_email = req_body.get("userEmail", "Not Provided")
        latitude = req_body.get("latitude")
        longitude = req_body.get("longitude")
        
        if not user_input:
            return jsonify({"error": "No input provided"}), 400

        # Location Resolution 
        location = "Location Not Provided by Client"
        if latitude is not None and longitude is not None:
            location = reverse_geocode(latitude, longitude)
            
        # Invoke the RAG chain
        response_text = retrieval_chain.invoke({
            "input": user_input,
            "user_name": user_name,
            "user_email": user_email
        })
        
        final_answer = response_text
        product_image_url = ""

        # Log data to Google Sheets (Runs in a separate thread)
        log_thread = threading.Thread(
            target=log_to_google_sheet, 
            args=([user_name, user_email, datetime.datetime.now().strftime("%d/%b/%Y %H:%M:%S"), location, user_input, final_answer],), 
            daemon=True
        )
        log_thread.start()

        return jsonify({"text": final_answer, "image_url": product_image_url})

    except Exception as e:
        print(f"\n!!!!!! AN ERROR OCCURRED IN /api/generate !!!!!!")
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred during generation."}), 500

# --- 7. Main Execution Block (Only for Local Testing) ---
if __name__ == '__main__':
    # This block is only for testing locally on your computer, Vercel ignores it.
    app.run(debug=True, port=5000, host='127.0.0.1')
