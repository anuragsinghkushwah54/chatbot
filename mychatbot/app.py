# --- Core Imports ---
import traceback
import threading
import time
import json # <-- Already correctly imported
import os
import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from operator import itemgetter 

# --- Google Sheets Imports ---
import gspread
from google.oauth2.service_account import Credentials

# --- LangChain Imports for RAG ---
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma

# OpenAI Classes
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# LCEL Components
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- External Modules ---
import fetch

# --- 1. Centralized Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAPS_API_KEY = os.getenv("MAPS_API_KEY")

if not OPENAI_API_KEY:
    print("FATAL ERROR: OPENAI_API_KEY not found. Set it as an environment variable.")
    exit()

# Google Sheets Configuration
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
SERVICE_ACCOUNT_FILE = 'D:/Desktop/Zalgochatbot/credentials.json' 
SHEET_NAME = "Zalgochatbot"

# --- 2. Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Reverse Geocoding Function (No Change) ---
def reverse_geocode(lat, lon):
    """Converts latitude and longitude to a detailed address using the geocode.maps.co API."""
    
    API_KEY = MAPS_API_KEY 

    if not API_KEY:
        print("WARNING: Geocoding skipped. API Key is not set.")
        return f"Lat: {lat}, Lon: {lon} (Geocoding API Key Missing)"
        
    base_url = "https://geocode.maps.co/reverse"
    params = {
        "lat": lat,
        "lon": lon,
        "api_key": API_KEY,
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if response.status_code == 200 and 'display_name' in data:
            formatted_address = data['display_name']
            return formatted_address
        elif response.status_code == 404:
            return f"Location Found: Lat: {lat}, Lon: {lon} (No Geocode Results)"
        else:
            return f"Geocode Error: API returned status {response.status_code}"
            
    except requests.exceptions.RequestException as e:
        print(f"Geocoding network error: {e}")
        return f"Lat: {lat}, Lon: {lon} (Geocode Failed)"
    except Exception as e:
        print(f"An unexpected error occurred during geocoding: {e}")
        return f"Lat: {lat}, Lon: {lon} (Unknown Error)"

# --- 3. Google Sheets Logging Function (No Change) ---
def log_to_google_sheet(user_data):
    """Appends a row of data to the specified Google Sheet."""
    try:
        creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        client = gspread.authorize(creds)
        sheet = client.open(SHEET_NAME).sheet1 
        sheet.append_row(user_data)
        print("Data successfully logged to Google Sheet!")
    except Exception as e:
        print(f"An error occurred while logging to Google Sheet: {e}")
        traceback.print_exc()

# --- 4. Background Data Fetching (No Change) ---
fetch_thread = threading.Thread(target=fetch.run_fetch_loop, daemon=True)
fetch_thread.start()

# --- 5. RAG Pipeline Setup (No Change) ---
retrieval_chain = None
retriever = None 
try:
    print("Waiting for initial data file to be created...")
    max_wait_time = 60
    wait_start = time.time()
    while not os.path.exists(fetch.OUTPUT_FILE):
        if time.time() - wait_start > max_wait_time:
            raise FileNotFoundError("Initial data file was not created within the timeout period.")
        time.sleep(2)
    print("Initial data file found. Proceeding with RAG setup.")

    loader = TextLoader(fetch.OUTPUT_FILE, encoding='utf-8')
    documents = loader.load()
    
    # IMPROVEMENT: Increased chunk size and overlap for better semantic context
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100) 
    docs = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = Chroma.from_documents(docs, embeddings)
    
    retriever = vector_store.as_retriever(
        search_kwargs={'k': 6} # Returns top 6 results regardless of score
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
    
    retrieval_chain = (
        {
            "context": itemgetter("input") | retriever | (lambda x: "\n\n".join(doc.page_content for doc in x)), 
            "input": itemgetter("input"),
            "user_name": itemgetter("user_name"),
            "user_email": itemgetter("user_email")
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    print("RAG pipeline setup complete.")
except Exception as e:
    print(f"FATAL ERROR during RAG setup: {e}")
    traceback.print_exc()
    exit()

# --- 6. Flask Route (Improved Image Extraction Logic) ---
@app.route("/api/generate", methods=["POST"])
def generate_api():
    print("\n--- Received a request at /api/generate ---")
    
    global retriever
    if retriever is None:
        print("Error: Retriever not initialized.")
        return jsonify({"error": "Retriever component missing. Check server logs."}), 500

    try:
        req_body = request.get_json()
        user_input = req_body.get("contents")
        user_name = req_body.get("userName", "User") 
        user_email = req_body.get("userEmail", "Not Provided")
        latitude = req_body.get("latitude")
        longitude = req_body.get("longitude")
        
        print(f"User input received: '{user_input}'")
        
        if not user_input:
            print("Error: No input provided by the user.")
            return jsonify({"error": "No input provided"}), 400

        # --- Location Resolution ---
        location = "Location Not Provided by Client"
        if latitude is not None and longitude is not None:
            location = reverse_geocode(latitude, longitude)
            print(f"Resolved Location: {location}")
        # ---------------------------

        # # 1. Retrieve documents first to check for image data (COMMENTED OUT)
        # print("Retrieving relevant documents for image extraction...")
        # retriever_docs = retriever.invoke(user_input) 
        
        product_image_url = "" # Initialize empty, as extraction is skipped.

        # # 2. Inspect the retrieved documents for a product and its image URL (COMMENTED OUT)
        # for doc in retriever_docs:
        #     print(f"DEBUG: Retrieved Document Content (Start): {doc.page_content[:200]}...")
            
        #     # --- Attempt 1: Parse as full WooCommerce JSON (from fetch.py fix) ---
        #     try:
        #         # Assuming the whole document chunk is the JSON string
        #         product_data = json.loads(doc.page_content)
                
        #         # Check for the key WooCommerce structure and non-empty images list
        #         if 'images' in product_data and isinstance(product_data['images'], list) and product_data['images']:
        #             extracted_url = product_data['images'][0].get('src')
                    
        #             if extracted_url:
        #                 product_image_url = extracted_url
        #                 print(f"✅ SUCCESS: Extracted Image URL (JSON): {product_image_url}")
        #                 break # Found the URL, stop checking documents
                        
        #     except json.JSONDecodeError:
        #         # --- Attempt 2: Check for Markdown Image Links (Fallback for general text) ---
        #         doc_content = doc.page_content.strip()
                
        #         # Checks for the pattern ![](URL) often created by html2text or markdown
        #         if doc_content.startswith('![](') and ')' in doc_content:
        #             # Simple parsing: find URL between first '(' and first ')'
        #             start_index = doc_content.find('(') + 1
        #             end_index = doc_content.find(')')
                    
        #             if start_index > 0 and end_index > start_index:
        #                 product_image_url = doc_content[start_index:end_index].strip()
        #                 if product_image_url:
        #                     print(f"✅ SUCCESS: Extracted Image URL (Markdown): {product_image_url}")
        #                     break
                
        #         continue
        #     except Exception as e:
        #         # Catch unexpected indexing/key errors if JSON structure is messy
        #         print(f"⚠️ Error during image URL extraction from doc: {e}")
        #         continue
        
        # 3. Invoke the RAG chain (LLM uses the retrieved context)
        print("Invoking the retrieval chain for LLM response...")
        response_text = retrieval_chain.invoke({
            "input": user_input,
            "user_name": user_name,
            "user_email": user_email
        })
        print("...Retrieval chain finished successfully.")
        
        final_answer = response_text

        # 4. Log data to Google Sheets
        current_time = datetime.datetime.now().strftime("%d/%b/%Y %H:%M:%S")
        row_data = [user_name, user_email, current_time, location, user_input, final_answer]
        log_thread = threading.Thread(target=log_to_google_sheet, args=(row_data,), daemon=True)
        log_thread.start()

        # 5. Return the final structured response (Text + Image URL)
        # Note: product_image_url will always be "" since the logic is commented out.
        return jsonify({
            "text": final_answer,
            "image_url": product_image_url
        })

    except Exception as e:
        print(f"\n!!!!!! AN ERROR OCCURRED IN /api/generate !!!!!!")
        print(f"Error details: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred. Check the server logs for details."}), 500

# --- 7. Main Execution Block ---
if __name__ == '__main__':
    app.run(debug=True, port=5000, host='127.0.0.1')
