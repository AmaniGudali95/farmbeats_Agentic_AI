# import sys
# import math
# sys.path.insert(0, ".")
# from sentence_transformers import SentenceTransformer
#
# def cosine(a, b):
#     dot = sum(x*y for x,y in zip(a,b))
#     na  = math.sqrt(sum(x*x for x in a))
#     nb  = math.sqrt(sum(x*x for x in b))
#     return dot / (na * nb)
#
# print("Loading SPECTER2...")
# #model = SentenceTransformer("allenai/specter2_base")
# model = SentenceTransformer("BAAI/bge-base-en-v1.5")
# print("Ready.")
#
# sentences = [
#     "corn needs water when soil is dry",
#     "irrigate maize at 50 percent field capacity",
#     "the weather forecast shows rain tomorrow",
#     "tractor maintenance and engine repair",
#     "stock market investment portfolio returns"
# ]
#
# v1, v2, v3, v4, v5 = model.encode(sentences, convert_to_numpy=True)
#
# print(f"\nSPECTER2 results:")
# print(f"corn vs maize:   {cosine(v1, v2):.3f}")
# print(f"corn vs weather: {cosine(v1, v3):.3f}")
# print(f"corn vs tractor: {cosine(v1, v4):.3f}")
# print(f"corn vs stocks:  {cosine(v1, v5):.3f}")



import sys
sys.path.insert(0, ".")
from rag_cli import AgriEmbedder, VectorStore, DB_PATH

embedder = AgriEmbedder()
store    = VectorStore(db_path=DB_PATH, embedder=embedder)

questions = [
    "What soil moisture triggers corn irrigation?",
    "What NDVI value indicates healthy crops?",
    "How does FarmBeats work without internet?",
]

for question in questions:
    print(f"\nQ: {question}")
    print("─" * 50)
    results = store.query(question, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"  [{i}] score={r['score']:.3f} — {r['text'][:120]}...")