import os
import sys
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from ddgs import DDGS
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool

load_dotenv()

# --- 0. ENVIRONMENT VALIDATION ---
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    sys.exit("Error: DEEPSEEK_API_KEY is not set in the environment or .env file.")

# --- 1. LOCAL TOOLS ---
@tool
def create_folder(folder_name: str) -> str:
    """Creates a new folder on the local file system."""
    try:
        os.makedirs(folder_name, exist_ok=True)
        return f"Folder '{folder_name}' created successfully."
    except Exception as e:
        return f"Failed to create folder '{folder_name}': {e}"

@tool
def create_file(file_name: str, content: str) -> str:
    """Creates or overwrites a file with the specified content, automatically creating missing parent directories."""
    try:
        folder = os.path.dirname(file_name)
        if folder:
            os.makedirs(folder, exist_ok=True)
            
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File '{file_name}' created successfully."
    except Exception as e:
        return f"Failed to create file '{file_name}': {e}"

# --- 2. WEB TOOLS ---
@tool
def web_search(query: str) -> str:
    """Searches the internet for information via DuckDuckGo.
    Returns titles, links, and snippets for top results."""
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3):
                results.append(
                    f"Title: {r.get('title', 'N/A')}\n"
                    f"Link: {r.get('href', 'N/A')}\n"
                    f"Content: {r.get('body', 'N/A')}\n"
                )
    except Exception as e:
        return f"Search service unavailable or failed: {e}"
    
    if not results:
        return "No relevant search results found."
    return "\n---\n".join(results)

@tool
def scrape_website(url: str) -> str:
    """Fetches text content from a specific HTTP/HTTPS URL. Useful to read a webpage in detail."""
    parsed_url = urlparse(url)
    if parsed_url.scheme not in ("http", "https"):
        return f"Security Error: Only HTTP and HTTPS URLs are allowed. Provided scheme: '{parsed_url.scheme}'"

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, timeout=10, headers=headers, allow_redirects=True)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Strip script, style, and navigation tags for cleaner context
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.extract()
            
        text = soup.get_text(separator=" ", strip=True)
        # Prevent context window flooding
        return text[:3500] + ("..." if len(text) > 3500 else "")
    except requests.exceptions.RequestException as e:
        return f"Network error while fetching '{url}': {e}"
    except Exception as e:
        return f"Error processing webpage content: {e}"

# Combine all tools into a unified list
tools = [create_folder, create_file, web_search, scrape_website]

# --- 3. AGENT INITIALIZATION & EXECUTION ---
llm = ChatDeepSeek(
    model="deepseek-chat", 
    api_key=api_key,
    temperature=0
)

system_prompt = (
    "You are an autonomous AI agent. You can search the internet "
    "and interact with the local file system. Keep your search queries "
    "short and concise (2-4 words) for best search results."
)

# Modern agent instantiation using LangChain's recommended create_agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)

if __name__ == "__main__":
    task = (
        "Research current trends in Agentic AI in 2026. "
        "Create a folder named 'Research' and save your findings "
        "in a file named 'agentic_ai_trends.txt'."
    )
    
    inputs = {"messages": [("user", task)]}
    
    # Stream execution states
    for chunk in agent.stream(inputs, stream_mode="values"):
        latest_message = chunk["messages"][-1]
        latest_message.pretty_print()