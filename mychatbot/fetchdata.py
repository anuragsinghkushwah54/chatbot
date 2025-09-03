# The requests module is used to send HTTP requests to the WordPress API.
import requests
# The html2text module is used to convert the HTML content of posts into clean, readable text.
import html2text

# --- Configuration ---
# This is the URL to your WordPress REST API endpoint for posts.
# - You can change `per_page=100` to fetch more or fewer posts at once (100 is the max).
# - If you have more than 100 posts, you would need to add logic to handle pagination (e.g., adding `&page=2`).
# - Remove `?search=life` if you want to fetch ALL posts instead of just those matching a search term.
API_URL = "https://bioage.pro/wp-json/wp/v2/posts?per_page=100"

# The name of the file where the cleaned data will be saved.
OUTPUT_FILE = "wordpress_data.txt"


def fetch_and_process_data():
    """
    Fetches post data from the WordPress API, cleans it, and saves it to a text file.
    """
    print(f"Attempting to fetch data from: {API_URL}")

    try:
        # Send a GET request to the API URL.
        response = requests.get(API_URL, timeout=30)
        # Raise an HTTPError if the HTTP request returned an unsuccessful status code (like 404 or 500).
        response.raise_for_status()
        
        # Parse the JSON response from the API into a Python list of dictionaries.
        posts = response.json()

        if not posts:
            print("Warning: The API returned no posts. The output file will be empty.")
            return

        # Initialize the html2text converter.
        h = html2text.HTML2Text()
        # Configure the converter to ignore links, as they are not needed for the chatbot's context.
        h.ignore_links = True

        # Open the output file in write mode with UTF-8 encoding to handle special characters.
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            # Loop through each post dictionary in the list.
            for post in posts:
                # Safely get the title and content, providing a default if they don't exist.
                title = post.get('title', {}).get('rendered', 'No Title')
                content_html = post.get('content', {}).get('rendered', '')

                # Convert the HTML content to clean, plain text.
                content_text = h.handle(content_html)
                
                # Write the formatted title and content to the file.
                f.write(f"## {title}\n\n")
                f.write(f"{content_text.strip()}\n\n")
                # Add a clear separator between posts for better parsing later.
                f.write("---\n\n")

        print(f"Success! Fetched and saved {len(posts)} posts to '{OUTPUT_FILE}'")

    except requests.exceptions.RequestException as e:
        # Handle network-related errors (e.g., DNS failure, refused connection, timeout).
        print(f"An error occurred during the network request: {e}")
    except Exception as e:
        # Handle other potential errors (e.g., JSON decoding error, file writing issues).
        print(f"An unexpected error occurred: {e}")


# --- Main Execution Block ---
# This ensures the fetch_and_process_data function is called only when the script is run directly.
if __name__ == "__main__":
    fetch_and_process_data()
