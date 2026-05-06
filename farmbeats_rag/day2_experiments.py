import pathlib, sys
sys.path.insert(0, ".")
from rag_cli import chunk_text

TEXT = pathlib.Path("docs/farmbeats_overview.txt").read_text()

def ex1_inspect():
    chunks = chunk_text(TEXT, "farmbeats", chunk_size=400, overlap=60)
    print(f"Total chunks: {len(chunks)}")
    print(f"Total words : {len(TEXT.split())}")
    print()
    for c in chunks:
        words = c["text"].split()
        print(f"── Chunk {c['chunk_index']} (words {c['word_start']}–{c['word_start']+len(words)}) ──")
        print(f"   First 20 words: {' '.join(words[:20])}")
        print(f"   Last  20 words: {' '.join(words[-20:])}")
        print()

def ex2_size_sweep():
    sizes = [100, 200, 400, 600, 800, 1200]
    print(f"{'Size':>6} {'Chunks':>7} {'Avg words':>10} {'Min':>6} {'Max':>6}")
    print("─" * 42)
    for size in sizes:
        overlap = size // 6
        chunks  = chunk_text(TEXT, "test", chunk_size=size, overlap=overlap)
        lengths = [len(c["text"].split()) for c in chunks]
        avg = sum(lengths) // len(lengths)
        print(f"{size:>6} {len(chunks):>7} {avg:>10} {min(lengths):>6} {max(lengths):>6}")

def ex3_overlap_test():
    SIZE = 400
    overlaps = [0, 40, 80, 120, 200]
    print(f"Chunk size fixed at {SIZE} words\n")
    print(f"{'Overlap':>8} {'Ratio':>7} {'Chunks':>8} {'Redundant words':>16}")
    print("─" * 44)
    for ov in overlaps:
        chunks   = chunk_text(TEXT, "test", chunk_size=SIZE, overlap=ov)
        stored   = sum(len(c["text"].split()) for c in chunks)
        original = len(TEXT.split())
        redundant = stored - original
        ratio = ov / SIZE
        print(f"{ov:>8} {ratio:>6.0%} {len(chunks):>8} {redundant:>16}")
    print()
    print("── Boundary at overlap=0 ──")
    chunks_0 = chunk_text(TEXT, "test", chunk_size=SIZE, overlap=0)
    print("End of chunk 0  :", " ".join(chunks_0[0]["text"].split()[-15:]))
    print("Start of chunk 1:", " ".join(chunks_0[1]["text"].split()[:15]))
    print()
    print("── Boundary at overlap=120 ──")
    chunks_120 = chunk_text(TEXT, "test", chunk_size=SIZE, overlap=120)
    print("End of chunk 0  :", " ".join(chunks_120[0]["text"].split()[-15:]))
    print("Start of chunk 1:", " ".join(chunks_120[1]["text"].split()[:15]))

print("=" * 50)
print("EXERCISE 1 — Inspect chunks")
print("=" * 50)
ex1_inspect()

print("=" * 50)
print("EXERCISE 2 — Size sweep")
print("=" * 50)
ex2_size_sweep()

print()
print("=" * 50)
print("EXERCISE 3 — Overlap test")
print("=" * 50)
ex3_overlap_test()
