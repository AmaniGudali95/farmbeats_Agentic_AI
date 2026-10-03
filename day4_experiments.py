import time
import math
import sys
sys.path.insert(0,".")
import chromadb
from rag_cli import AnthropicEmbedder,VectorStore,DB_PATH


def ex1_inspect_db():
    embedder=AnthropicEmbedder()
    store=VectorStore(db_path=DB_PATH, embedder=embedder)
    print(f"total chunks stored:{store.count()}")

    result=store.collection.get(limit=1, include=["documents","metadatas","embeddings"])
    print(f"first chunk id:{result['ids'][0]}")
    print(f"first chunk metadata:{result['metadatas'][0]}")
    print(f"first chunk text:{result['documents'][0][:100]}")
    print(f"first chunk vector (first 5 numbers):{result['embeddings'][0][:5]}")

ex1_inspect_db()

def ex2_manual_query():
    embedder=AnthropicEmbedder()
    store=VectorStore(db_path=DB_PATH,embedder=embedder)
    question="What soil moisture level triggers irrigation for corn?"
    results=store.query(question,top_k=3)

    print(f"Query:{question}")
    print()
    for i,r in enumerate(results,1):
        print(f"Result {i} - store: {r['score']:.3f}")
        print(f"Source:{r['source']}")
        print(f"Text:{r['text'][:100]}")
        print()
ex2_manual_query()

def ex3_speed_test():
    embedder=AnthropicEmbedder()
    store=VectorStore(db_path=DB_PATH,embedder=embedder)

    question="soil moisture irrigation corn"
    query_vector=embedder.embed([question])[0]

    start=time.time()
    for _ in range(100):
        store.collection.query(query_embeddings=[query_vector],
                               n_results=3)
    chroma_time=time.time()-start
    print(f"Chromadb-100 queries in {chroma_time:.3f}s")

    all_chunks=store.collection.get(include=["embeddings","documents"])
    vectors=all_chunks["embeddings"]

    def cosine(a,b):
        dot=sum(x*y for x,y in zip(a,b))
        na=math.sqrt(sum(x*x for x in a))
        nb=math.sqrt(sum(x*x for x in b))
        return dot/(na*nb)

    start=time.time()
    for _ in range(100):
        scores=[(cosine(query_vector,v),i) for i,v in enumerate(vectors)]
        scores.sort(reverse=True)
        top3=scores[:3]
    brute_time=time.time()-start
    print(f"Brute force - 100 queries in {brute_time:.3f}s")
    print(f"ChromaDB is {brute_time/chroma_time:.1f}x faster")
ex3_speed_test()

def ex4_wrong_embedder():
    import random
    embedder=AnthropicEmbedder()
    store=VectorStore(db_path=DB_PATH,embedder=embedder)
    random_vector=[random.uniform(-1,1) for _ in range(1536)]

    result=store.collection.query(query_embeddings=[random_vector],
                                  n_results=3,
                                  include=["documents","distances"])
    print("Results using a random vector:")
    for doc,dist in zip(result["documents"][0], result["distances"][0]):
        score=round(1-dist,3)
        print(f"score={score:.3f} {doc[:80]}")

ex4_wrong_embedder()