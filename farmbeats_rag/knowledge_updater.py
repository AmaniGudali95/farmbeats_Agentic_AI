import requests
import hashlib
import pathlib
import sys
sys.path.insert(0, ".")
from rag_cli import AnthropicEmbedder, VectorStore, DB_PATH, chunk_text


def search_new_papers(query="precision agriculture IoT sensors", limit=5):
    """
    Search Semantic Scholar for recent agriculture papers.
    Returns list of papers with title, abstract, and PDF url if available.
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,year,openAccessPdf,externalIds"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        papers = []

        for paper in data.get("data", []):
            papers.append({
                "title":    paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "year":     paper.get("year", 0),
                "pdf_url":  paper.get("openAccessPdf", {}).get("url", None),
                "paper_id": paper.get("externalIds", {}).get("DOI",
                                                             paper.get("paperId", ""))
            })

        return papers

    except Exception as e:
        print(f"Search failed: {e}")
        return []