# The requests module is used to send HTTP requests to the WordPress API.
import requests
# The html2text module is used to convert the HTML content of posts into clean, readable text.
import html2text
# The time module is used to pause the script for a specified duration.
import time
# The os module is used to check for the existence of files.
import os
# The json module is essential for serializing/deserializing the WooCommerce data.
import json

# --- Configuration ---
# List of all API URLs to fetch data from.
API_URLS = [
    # Standard WordPress Pages/Posts
    "https://bioage.pro/wp-json/wp/v2/pages?per_page=100", 
    # WooCommerce Products API (Structured Pricing/Image Data)
    "https://bioage.pro/wp-json/wc/v3/products?consumer_key=ck_0e7736538c89c52811a52d86b43617c0bfd0fe4a&consumer_secret=cs_b8d119d36b4f37570e9a33babc467fe1423b91ef" 
]

# The name of the file where the cleaned data will be saved.
OUTPUT_FILE = "wordpress_data.txt"

# --- Configuration for Repetitive Fetching ---
# The number of seconds to wait between each data fetch.
FETCH_INTERVAL_SECONDS = 600

# Separator used to denote the start/end of a data chunk for the RAG splitter.
# Triple hash with whitespace is highly recommended for clear segmentation.
CHUNK_SEPARATOR = "\n\n### ---\n\n"


def fetch_and_process_url(url, h):
    """
    Fetches data from a single WordPress/WooCommerce API URL and returns a list 
    of processed text strings.
    """
    print(f"  -> Fetching data from: {url}")
    processed_data = []
    
    try:
        # Send a GET request to the API URL.
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Parse the JSON response from the API into a Python list of dictionaries.
        items = response.json()
        
        if not items:
            print(f"  Warning: The API for {url} returned no items.")
            return []

        # --- Conditional Processing for WooCommerce Products ---
        if 'wc/v3/products' in url:
            for item in items:
                # 📢 FIX: Instead of extracting individual fields, save the 
                # entire JSON object as a string. This makes it parsable 
                # by Flask when retrieving the image URL.
                
                # Convert the Python dictionary back to a JSON string
                json_string = json.dumps(item) 
                
                # The full JSON object forms the document chunk
                formatted_item = json_string + CHUNK_SEPARATOR
                processed_data.append(formatted_item)

            print(f"  -> Success! Fetched and processed {len(items)} WooCommerce Products from {url}")
            return processed_data

        # --- Standard WordPress Post/Page Processing (Markdown Text) ---
        else:
            for item in items:
                title = item.get('title', {}).get('rendered', 'No Title')
                content_html = item.get('content', {}).get('rendered', '')
                
                # Convert HTML to clean Markdown text.
                content_text = h.handle(content_html)
                content_text = content_text.replace('`', '') 
                
                # Format the text content for RAG, ensuring a clean title separation
                formatted_item = (
                    f"## {title}\n\n"
                    f"{content_text.strip()}"
                    f"{CHUNK_SEPARATOR}" # Separator between items
                )
                processed_data.append(formatted_item)

            print(f"  -> Success! Fetched and processed {len(items)} standard items from {url}")
            return processed_data

    except requests.exceptions.RequestException as e:
        print(f"  An error occurred during network request for {url}: {e}")
        return []
    except Exception as e:
        print(f"  An unexpected error occurred for {url}: {e}")
        return []


def fetch_and_process_all_data():
    """
    Coordinates fetching data from all configured API URLs, cleans it, 
    and saves the combined result to a text file.
    Returns True if successful, False otherwise.
    """
    print("\n--- Starting Data Fetch Cycle ---")
    
    # Initialize the html2text converter once.
    h = html2text.HTML2Text()
    # Ignoring links in regular WP content keeps chunks cleaner and LLM focused.
    h.ignore_links = True 
    
    all_data = []
    total_items = 0
    
    # Loop through each configured URL and fetch/process the data.
    for url in API_URLS:
        data = fetch_and_process_url(url, h)
        if data: 
            all_data.extend(data)
            total_items += len(data)
            
    if not all_data:
        print("Failure: Could not retrieve any data from the configured URLs.")
        return False

    # Save all the combined and processed data to the output file.
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("".join(all_data))
            
        print(f"\nSuccess! Combined and saved a total of {total_items} items to '{OUTPUT_FILE}'")
        return True

    except Exception as e:
        print(f"\nAn error occurred while writing to the file: {e}")
        return False


def run_fetch_loop():
    """
    Continuously fetches data in a loop.
    """
    print("Starting background data fetch task...")
    while True:
        fetch_and_process_all_data()
        print(f"\nWaiting for {FETCH_INTERVAL_SECONDS} seconds before the next fetch...")
        time.sleep(FETCH_INTERVAL_SECONDS)


# This ensures the function is only called when the script is run directly.
if __name__ == "__main__":
    run_fetch_loop()
