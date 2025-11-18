# --- Core Imports ---
import traceback
import threading
import os
import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from operator import itemgetter 
#import json # ADDED: Required to parse JSON credentials from environment variable

# --- Google Sheets Imports ---
#import gspread
#from google.oauth2.service_account import Credentials

# --- LangChain/Qdrant Imports for RAG (FAST COLD START) ---
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_qdrant import QdrantVectorStore 

# --- 1. Centralized Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAPS_API_KEY = os.getenv("MAPS_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# NEW: Read JSON credentials securely from Vercel Environment Variable
#GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_SHEETS_CREDENTIALS") 

#COLLECTION_NAME = "zalgo-rag-collection"
#SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
#SHEET_NAME = "Zalgochatbot"

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

# --- 3. Google Sheets Logging Function (UPDATED for Vercel) ---
'''def log_to_google_sheet(user_data):
    """Appends a row of data to the specified Google Sheet using ENV credentials."""
    # Check if credentials are set before attempting to log
    if not GOOGLE_CREDENTIALS_JSON:
        print("WARNING: Google Sheets logging skipped (Credentials not found).")
        return

    try:
        # Load credentials directly from the JSON string stored in the environment variable
        creds_info = json.loads(GOOGLE_CREDENTIALS_JSON)
        creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
        
        client = gspread.authorize(creds)
        sheet = client.open(SHEET_NAME).sheet1 
        sheet.append_row(user_data)
        
    except Exception as e:
        print(f"FATAL Google Sheet Logging Error: {e}")
        traceback.print_exc()'''

# --- 5. RAG Pipeline Setup (SIMPLIFIED FOR FAST COLD START) ---
retrieval_chain = None
retriever = None 
try:
    print("RAG Setup: Initializing clients and connecting to existing Qdrant collection...")
    
    # 1. Initialize Clients (Fast connections to external services)
    qdrant_client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY) 

    # 2. Connect to Existing Collection (NO data loading/upserting here)
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
    retriever = None 
    traceback.print_exc()

# --- 6. Flask Routes ---

@app.route("/", methods=["GET"])
def home():
    """Simple root route to confirm the server is running."""
    if retriever is None:
        return jsonify({"status": "error", "message": "RAG system failed to initialize. Check logs."}), 500
    return jsonify({"status": "ok", "message": "Chatbot API is operational. Send POST request to /api/generate."})


@app.route("/api/generate", methods=["POST"])
def generate_api():
    
    global retriever
    if retriever is None:
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
       ''' log_thread = threading.Thread(
            target=log_to_google_sheet, 
            args=([user_name, user_email, datetime.datetime.now().strftime("%d/%b/%Y %H:%M:%S"), location, user_input, final_answer],), 
            daemon=True
        )
        log_thread.start()'''

        return jsonify({"text": final_answer, "image_url": product_image_url})

    except Exception as e:
        print(f"\n!!!!!! AN ERROR OCCURRED IN /api/generate !!!!!!")
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred during generation."}), 500

# --- 7. Main Execution Block (Only for Local Testing) ---
if __name__ == '__main__':
    app.run(debug=True, port=5000, host='127.0.0.1')

