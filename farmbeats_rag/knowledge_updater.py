import requests
import hashlib
import sys
import schedule
import time
from datetime import datetime
from rag_cli import AnthropicEmbedder, AgriEmbedder, VectorStore, DB_PATH, chunk_text
import pypdf
import io

TRUSTED_SOURCES = [
    {
        "url": "https://www.fao.org/3/y4263e/y4263e06.htm",
        "name": "fao_irrigation_management"
    },
    {
        "url": "https://extension.umn.edu/growing-guide/corn",
        "name": "umn_corn_guide"
    },
    {
        "url": "https://www.nrcs.usda.gov/conservation-basics/natural-resource-concerns/soil/soil-health",
        "name": "usda_soil_health"
    },
    {
        "url": "https://www.fao.org/3/y4263e/y4263e07.htm",
        "name": "fao_crop_water_requirements"
    },
]

def search_new_papers(query="precision agriculture IoT sensors", limit=5):
    url="https://api.semanticscholar.org/graph/v1/paper/search"
    params={
        "query": query,
        "limit": limit,
        "fields": "title,abstract,year,openAccessPdf,externalIds"
    }
    try:
        response = requests.get(url,params=params, timeout=10)
        data=response.json()
        papers = []
        for paper in data.get("data",[]):
            papers.append({
                "title": paper.get("title",""),
                "abstract": paper.get("abstract",""),
                "year": paper.get("year",0),
                "pdf_url": paper.get("openAccessPdf",{}).get("url", None),
                "paper_id": paper.get("externalIds",{}).get("DOI", paper.get("paperId",""))
            })
        return papers
    except Exception as e:
        print(f"Search failed:{e}")
        return []

def paper_already_indexed(store,paper_id):
    try:
        chunk_id=(
            f"semantic_scholar__"
            f"{hashlib.md5(paper_id.encode()).hexdigest()[:8]}"
            f"__chunk_0"
        )
        results=store.collection.get(ids=[chunk_id])
        return len(results["ids"])>0
    except Exception:
        return False

def download_pdf_text(pdf_url):
    try:
        response=requests.get(pdf_url, timeout=30)
        if response.status_code!=200:
            return None
        pdf_file=io.BytesIO(response.content)
        reader = pypdf.PdfReader(pdf_file)

        pages=[]
        for i,page in enumerate(reader.pages):
            text=page.extract_text()
            if text and text.strip():
                pages.append(f"[Page {i+1}]\n{text.strip()}")
        return "\n\n".join(pages)
    except Exception as e:
        print(f"PDF download failed:{e}")
        return None

def add_paper_to_index(store,paper):
    if not paper["abstract"] and not paper["pdf_url"]:
        print(f"Skipping '{paper['title'][:50]}' - no content")
        return False

    full_text=None
    if paper["pdf_url"]:
        print(f"Downloading full PDF...")
        full_text=download_pdf_text(paper["pdf_url"])

    if full_text:
        text=f"Title: {paper['title']}\n\n{full_text}"
        source=(
            f"semantic_scholar__"
            f"{hashlib.md5(paper['paper_id'].encode()).hexdigest()[:8]}"
            f"__full"
        )
        print(f"Indexing full paper ({len(full_text.split())} words)")
    else:
        if not paper["abstract"]:
            print(f" Skipping '{paper['title'][:50]}' - no abstract or PDF")
            return False

        text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
        source = (
            f"semantic_scholar__"
            f"{hashlib.md5(paper['paper_id'].encode()).hexdigest()[:8]}"
            f"__abstract"
        )
        print(f" Falling back to abstract only")
    chunks=chunk_text(text, source=source, chunk_size=400, overlap=60)

    if not chunks:
        print(f" Skipping '{paper['title'][:50]}' - too short to chunk")
        return False

    store.add_chunks(chunks)
    print(f" Added '{paper['title'][:60]}' ({len(chunks)} chunks)")
    return True
def source_already_indexed(store, source_name):
    """
    Check if a web source is already in ChromaDB.
    Uses source name as identifier.
    """
    try:
        chunk_id = f"{source_name}__chunk_0"
        results = store.collection.get(ids=[chunk_id])
        return len(results["ids"]) > 0
    except Exception:
        return False


def scrape_agricultural_source(store, url, source_name):
    """
    Fetches content from a trusted agricultural website.
    Extracts main text, chunks it, adds to ChromaDB.
    """
    try:
        from bs4 import BeautifulSoup

        print(f"  Fetching {source_name}...")
        response = requests.get(url, timeout=15)

        if response.status_code != 200:
            print(f"  Failed — status {response.status_code}")
            return False

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove navigation, headers, footers, scripts
        for tag in soup(['nav', 'header', 'footer',
                         'script', 'style', 'aside']):
            tag.decompose()

        text = soup.get_text(separator=' ', strip=True)

        # Clean up excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text.split()) < 100:
            print(f"  Too short — skipping {source_name}")
            return False

        chunks = chunk_text(
            text,
            source=source_name,
            chunk_size=400,
            overlap=60
        )

        if not chunks:
            print(f"  No chunks produced — skipping {source_name}")
            return False

        store.add_chunks(chunks)
        print(f"  Added {source_name} ({len(chunks)} chunks)")
        return True

    except Exception as e:
        print(f"  Failed to scrape {source_name}: {e}")
        return False


def update_web_sources(store):
    """
    Fetches and indexes content from trusted agricultural websites.
    Skips sources already in the index.
    """
    print(f"\n{'─'*60}")
    print("Updating from trusted web sources...")
    print(f"{'─'*60}")

    added   = 0
    skipped = 0

    for source in TRUSTED_SOURCES:
        if source_already_indexed(store, source["name"]):
            print(f"  Already indexed: {source['name']}")
            skipped += 1
        else:
            success = scrape_agricultural_source(
                store,
                source["url"],
                source["name"]
            )
            if success:
                added += 1

    print(f"\nWeb sources: {added} added, {skipped} skipped")
    return added
def update_knowledge_base():
    print(f"\n{'='*60}")
    print(f"Knowledge base update - "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"
          )
    print(f"{'='*60}")

    #embedder = AnthropicEmbedder()
    embedder=AgriEmbedder()
    store = VectorStore(db_path=DB_PATH, embedder=embedder)

    print(f"Current index size: {store.count()} chunks")

    queries = [
        "precision agriculture IoT soil moisture sensors",
        "NDVI crop health satellite imagery farming",
        "FarmBeats data driven agriculture smallholder",
        "irrigation scheduling machine learning crops",
    ]
    added=0
    skipped=0

    for query in queries:
        print(f"\nSearching: '{query}'")
        papers = search_new_papers(query, limit=3)
        print(f"Found {len(papers)} papers")

        for paper in papers:
            if not paper["paper_id"]:
                continue

            if paper_already_indexed(store, paper["paper_id"]):
                print(f" Already indexed: '{paper['title'][:50]}'")
                skipped+=1
                continue

            success=add_paper_to_index(store,paper)
            if success:
                added+=1
    web_added = update_web_sources(store)
    added += web_added
    print(f"\nupdate complete:")
    print(f" added: {added} new papers")
    print(f"Skipped: {skipped} duplicates")
    print(f" total chunks now: {store.count()}")


def run_scheduler():
    print("Knowledge updater started — runs every Monday at 06:00")
    schedule.every().monday.at("06:00").do(update_knowledge_base)
    while True:
        schedule.run_pending()
        time.sleep(3600)

if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="--schedule":
        run_scheduler()
    else:
        update_knowledge_base()
