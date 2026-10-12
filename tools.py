from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from rich import print
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# Initialize Tavily
tavily = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


# ============================================================
# WEB SEARCH TOOL
# ============================================================

@tool
def web_search(query: str) -> str:
    """
    Search the web for recent and reliable information
    on a given topic and return titles, URLs, and snippets.
    """

    results = tavily.search(
        query=query,
        max_results=5
    )

    out = []

    for r in results["results"]:
        out.append(
            f'Title: {r["title"]}\n'
            f'URL: {r["url"]}\n'
            f'Snippet: {r["content"][:300]}\n'
        )

    return "\n---\n".join(out)


# ============================================================
# URL SCRAPER TOOL
# ============================================================

@tool
def scrape_url(url: str) -> str:
    """
    Scrape and return clean text content from a given URL
    for deeper reading.
    """

    try:

        response = requests.get(
            url,
            timeout=8,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Remove unnecessary HTML elements
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        # Extract clean text
        text = soup.get_text(
            separator=" ",
            strip=True
        )

        return text[:3000]

    except Exception as e:

        return f"Could not scrape URL: {str(e)}"