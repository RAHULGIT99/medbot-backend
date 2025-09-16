import os
import httpx
from dotenv import load_dotenv
import random

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from pinecone import Pinecone

# --- Environment Variables ---
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")


# Remote Hugging Face Space Embedding API URL
EMBEDDING_API_URL = "https://rahulbro123-mymodel.hf.space/embed"

if not (PINECONE_API_KEY and PINECONE_INDEX_NAME):
    raise RuntimeError("Missing pinecone environment variables.")


numbers = [1, 2, 3]

# Pick a random element
selected = random.choice(numbers)

# Build your key
api_key = "GROQ_API_KEY_" + str(selected)

print(api_key)
GROQ_API_KEY = os.getenv(api_key)
print(GROQ_API_KEY)
app = FastAPI()

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# --- Pinecone Client ---
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# --- Embeddings Helper (Async) ---
async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed texts using the remote Hugging Face API."""
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                EMBEDDING_API_URL,
                json={"texts": texts},
            )
            response.raise_for_status()  # Raise an exception for bad status codes
            return response.json()["embeddings"]
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Embedding API error: {e.response.text}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

# --- Request Schema ---
class QueryRequest(BaseModel):
    question: str

# --- Helpers ---
async def rephrase_query(question: str) -> str:
    """Use Groq LLM to rephrase user query."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert in medical query understanding. "
                    "see what the user is asking and rephrase it into a concise, specific query that llm can use to find relevant information in a medical document. "
                    "Your task is to rephrase the user's question into a concise, "
                    "specific query that is likely to be found in a medical document. "
                    "IMPORTANT: Only return the rephrased query, without any of your own conversational text or explanation."
                ),
            },
            {"role": "user", "content": question},
        ],
        "temperature": 0,
    }

    async with httpx.AsyncClient() as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code != 200:
            error_detail = r.json()
            raise HTTPException(
                status_code=r.status_code,
                detail=f"Groq rephrasing failed: {error_detail}",
            )
        return r.json()["choices"][0]["message"]["content"].strip()

async def retrieve_context(query_vector: List[float], top_k: int = 5) -> str:
    """Search Pinecone for relevant context."""
    try:
        results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        if not results.matches:
            return ""
        return "\n\n---\n\n".join([m["metadata"]["text"] for m in results.matches])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context retrieval failed: {e}")

async def get_medical_answer(query: str, context: str) -> str:
    """Use Groq LLM to answer medical question based on context."""
    if not context.strip():
        return "I can't tell you, please contact the nearest doctor."

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful Medical Assistant speaking directly to a patient. "
                    "Prescribe medicines/tablets if there in the context."
                    "Answer the user's question based ONLY on the provided context. "
                    "Address the user directly (e.g., 'You should...'). "
                    "If the answer is not in the context, say: "
                    "'I can't tell you, please contact the nearest doctor.'"
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        "temperature": 0.2,
    }

    async with httpx.AsyncClient() as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code != 200:
            error_detail = r.json()
            raise HTTPException(
                status_code=r.status_code,
                detail=f"Groq answering failed: {error_detail}",
            )
        return r.json()["choices"][0]["message"]["content"].strip()


import uvicorn
# --- API Endpoints ---
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Medical chatbot API is running"}

@app.post("/medical-chat")
async def medical_chat(request: QueryRequest) -> Dict[str, Any]:
    try:
        # Step 1: Rephrase
        rewritten_query = await rephrase_query(request.question)

        # Step 2: Embed using the custom API
        query_vector = (await embed_texts([rewritten_query]))[0]

        # Step 3: Retrieve docs
        context = await retrieve_context(query_vector, top_k=5)

        # Step 4: Answer
        answer = await get_medical_answer(rewritten_query, context)

        return {
            "original_query": request.question,
            "rewritten_query": rewritten_query,
            "answer": answer,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Run Server on Port 6000 ---
