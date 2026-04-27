import math
import sys
sys.path.insert(0,".")
from rag_cli import AnthropicEmbedder

def cosine_similarity(a,b):
    dot_prod=sum(x*y for x,y in zip(a,b))
    len_a=math.sqrt(sum(x*x for x in a))
    len_b=math.sqrt(sum(x*x for x in b))
    return dot_prod/(len_a*len_b)

vec1 = [1, 0, 0]
vec2 = [1, 0, 0]
vec3 = [0, 1, 0]

print(cosine_similarity(vec1, vec2))
print(cosine_similarity(vec1, vec3))

embedder=AnthropicEmbedder()
sentences=["corn needs water when soil is dry",
           "irrigate maize at 50 percent field capacity",
           "the weather forecast shows rain tomorrow"]
vectors=embedder.embed(sentences)
sim_1=cosine_similarity(vectors[0],vectors[1])
sim_2=cosine_similarity(vectors[0],vectors[2])

print(f"corn vs maize: {sim_1:.3f}")
print(f"corn vs weather:{sim_2:.3f}")
