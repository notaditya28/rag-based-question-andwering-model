"""
RAG (Retrieval-Augmented Generation) System
Topic: Building RAG Agents with LLMs
"""

import math
import re
import os
from collections import Counter
from groq import Groq

# ── KNOWLEDGE BASE ──────────────────────────────────────
DOCUMENTS = [
    {
        "id": "doc1",
        "title": "Introduction to Neural Networks",
        "text": (
            "Neural networks are computational models inspired by the human brain. "
            "They consist of layers of interconnected nodes called neurons. "
            "Each neuron receives inputs, applies a weight and bias, then passes the result "
            "through an activation function. Deep learning refers to neural networks with many layers. "
            "Common activation functions include ReLU, sigmoid, and tanh."
        ),
    },
    {
        "id": "doc2",
        "title": "Backpropagation Explained",
        "text": (
            "Backpropagation is the algorithm used to train neural networks. "
            "It computes gradients of the loss function with respect to each weight "
            "using the chain rule of calculus. Gradients flow backward from the output layer "
            "to the input layer. The optimizer such as SGD or Adam then updates weights "
            "to minimize the loss. Learning rate controls the step size during updates."
        ),
    },
    {
        "id": "doc3",
        "title": "Large Language Models Overview",
        "text": (
            "Large Language Models (LLMs) are transformer-based neural networks trained on "
            "massive text datasets. They learn to predict the next token in a sequence. "
            "Popular LLMs include GPT-4, Claude, and Gemini. LLMs can perform tasks like "
            "summarization, translation, code generation, and question answering. "
            "Prompt engineering is key to getting high-quality outputs from LLMs."
        ),
    },
    {
        "id": "doc4",
        "title": "Retrieval-Augmented Generation (RAG)",
        "text": (
            "RAG combines information retrieval with text generation. A retriever fetches "
            "relevant documents from a knowledge base given a query. These documents are "
            "provided as context to an LLM, which then generates a grounded answer. "
            "RAG reduces hallucination and allows LLMs to access up-to-date information "
            "without retraining. Vector databases like FAISS or Pinecone are commonly used."
        ),
    },
    {
        "id": "doc5",
        "title": "Transformer Architecture",
        "text": (
            "The Transformer architecture was introduced in the paper Attention is All You Need. "
            "It relies on self-attention mechanisms to capture relationships between tokens "
            "regardless of their distance. Transformers consist of encoder and decoder blocks, "
            "each containing multi-head attention and feed-forward layers. "
            "Positional encodings are added to preserve word order information."
        ),
    },
]

# ── CHUNKING ─────────────────────────────────────────────
def chunk_document(doc, chunk_size=2):
    sentences = re.split(r'(?<=[.!?])\s+', doc["text"].strip())
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk_text = " ".join(sentences[i:i + chunk_size])
        chunks.append({
            "doc_id": doc["id"],
            "doc_title": doc["title"],
            "chunk_id": f"{doc['id']}_c{i // chunk_size}",
            "text": chunk_text,
        })
    return chunks

def build_knowledge_base(documents):
    chunks = []
    for doc in documents:
        chunks.extend(chunk_document(doc))
    return chunks

# ── TF-IDF RETRIEVER ─────────────────────────────────────
def tokenize(text):
    return re.findall(r'\b[a-z]+\b', text.lower())

def compute_tf(tokens):
    count = Counter(tokens)
    total = len(tokens)
    return {word: cnt / total for word, cnt in count.items()}

def compute_idf(chunks):
    N = len(chunks)
    doc_freq = Counter()
    for chunk in chunks:
        unique = set(tokenize(chunk["text"]))
        for word in unique:
            doc_freq[word] += 1
    return {word: math.log(N / (1 + df)) for word, df in doc_freq.items()}

def tfidf_vector(tokens, idf):
    tf = compute_tf(tokens)
    return {word: tf_val * idf.get(word, 0) for word, tf_val in tf.items()}

def cosine_similarity(vec_a, vec_b):
    dot = sum(vec_a.get(w, 0) * vec_b.get(w, 0) for w in vec_b)
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)

def retrieve(query, chunks, idf, top_k=3):
    query_tokens = tokenize(query)
    query_vec = tfidf_vector(query_tokens, idf)
    scores = []
    for chunk in chunks:
        chunk_tokens = tokenize(chunk["text"])
        chunk_vec = tfidf_vector(chunk_tokens, idf)
        score = cosine_similarity(query_vec, chunk_vec)
        scores.append((score, chunk))
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_k]

# ── LLM GENERATION (Groq) ────────────────────────────────
def call_llm(system_prompt, user_message):
    client = Groq(api_key="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")
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return response.choices[0].message.content

# ── RAG PIPELINE ─────────────────────────────────────────
def rag_answer(query, chunks, idf, top_k=3):
    retrieved = retrieve(query, chunks, idf, top_k=top_k)
    context_parts = []
    for rank, (score, chunk) in enumerate(retrieved, 1):
        context_parts.append(
            f"[{rank}] (Source: {chunk['doc_title']}, score={score:.3f})\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)
    system_prompt = (
        "You are a knowledgeable AI assistant. Answer the user's question using ONLY "
        "the provided context. If the context does not contain enough information, say so. "
        "Be concise and accurate."
    )
    user_message = f"Context:\n{context}\n\nQuestion: {query}"
    answer = call_llm(system_prompt, user_message)
    return {
        "query": query,
        "retrieved_chunks": retrieved,
        "answer": answer,
    }

# ── MAIN ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  RAG System — Building RAG Agents with LLMs")
    print("=" * 60)

    kb = build_knowledge_base(DOCUMENTS)
    idf = compute_idf(kb)
    print(f"\n[INFO] Knowledge base ready: {len(kb)} chunks from {len(DOCUMENTS)} documents\n")

    queries = [
        "How does backpropagation work in neural networks?",
        "What is RAG and how does it reduce hallucination?",
        "What is the transformer architecture and how does attention work?",
    ]

    for q in queries:
        print(f"QUERY: {q}")
        print("-" * 50)
        result = rag_answer(q, kb, idf)
        print("TOP RETRIEVED CHUNKS:")
        for rank, (score, chunk) in enumerate(result["retrieved_chunks"], 1):
            print(f"  [{rank}] {chunk['doc_title']} (score={score:.3f})")
        print(f"\nLLM ANSWER:\n{result['answer']}")
        print("\n" + "=" * 60 + "\n")
        