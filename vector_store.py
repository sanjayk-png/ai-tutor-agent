import faiss
import numpy as np
import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("AIzaSyBIr_a1WMbblgGamDQaY2vVKnuLb2vzfGM"))

EMBED_MODEL = "models/text-embedding-004"

dimension = 768
index = faiss.IndexFlatL2(dimension)
vector_texts = []

def embed_text(text: str):
    result = genai.embed_content(model=EMBED_MODEL, content=text)
    return np.array(result["embedding"], dtype=np.float32)

def add_to_index(text: str):
    emb = embed_text(text)
    index.add(np.array([emb]))
    vector_texts.append(text)

def search_similar(query, k=5):
    q_emb = embed_text(query)
    distances, neighbors = index.search(np.array([q_emb]), k)
    results = []
    for i, idx in enumerate(neighbors[0]):
        results.append((vector_texts[idx], float(distances[0][i])))
    return results
