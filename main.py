# sarvam_pinecone_backend.py

import os
import httpx
from dotenv import load_dotenv
import random
from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from typing import List, Dict, Any
from pinecone import Pinecone
import requests
import base64
from typing import Optional

# --- Load environment variables ---
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not (PINECONE_API_KEY and PINECONE_INDEX_NAME):
    raise RuntimeError("Missing pinecone environment variables.")

numbers = [1, 2, 3]
selected = random.choice(numbers)
api_key = "GROQ_API_KEY_" + str(selected)
GROQ_API_KEY = os.getenv(api_key)

EMBEDDING_API_URL = "https://rahulbro123-embedding-model.hf.space/get_embeddings"

SARVAM_API_KEY = "sk_f8fjoda1_s83hQcvwfwwmPIwImLdTaReh"
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text" 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pinecone Client ---
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# --- Request/Response Models ---
class QueryRequest(BaseModel):
    question: str

# -------------------
# Sarvam TTS/STT Endpoints
# -------------------
@app.get("/tts")
def tts(text: str, language_code: str = "en-IN", speaker: str = "anushka"):
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": [text],
        "target_language_code": language_code,
        "speaker": speaker,
        "audio_format": "mp3"
    }

    response = requests.post(SARVAM_TTS_URL, headers=headers, json=payload)

    if response.status_code != 200:
        return JSONResponse(
            status_code=response.status_code,
            content={"error": response.text}
        )

    data = response.json()
    if "audios" in data and len(data["audios"]) > 0:
        audio_bytes = base64.b64decode(data["audios"][0])
        return Response(content=audio_bytes, media_type="audio/mpeg")

    return JSONResponse({"error": "No audio returned from Sarvam"})

@app.post("/stt")
async def stt_sarvam(file: UploadFile = File(...), language_code: Optional[str] = Query(None)):
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty audio file")

    headers = {"api-subscription-key": SARVAM_API_KEY}
    data = {"model": "saarika:v2.5"}
    if language_code:
        data["language_code"] = language_code

    files = {"file": (file.filename, contents, file.content_type)}

    resp = requests.post(SARVAM_STT_URL, headers=headers, data=data, files=files)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"STT error: {resp.text}")

    resp_json = resp.json()
    transcript = resp_json.get("transcript") or resp_json.get("text") or ""
    return JSONResponse({"transcript": transcript})

# -------------------
# Embedding & Retrieval
# -------------------
async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed texts using the remote Hugging Face API."""
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(EMBEDDING_API_URL, json={"texts": texts})
            response.raise_for_status()
            return response.json()["embeddings"]
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Embedding API error: {e.response.text}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

async def retrieve_context(query_vector: List[float], top_k: int = 5) -> str:
    """Search Pinecone for relevant context."""
    try:
        results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        if not results.matches:
            return ""
        return "\n\n---\n\n".join([m["metadata"]["text"] for m in results.matches])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context retrieval failed: {e}")

async def get_answer(query: str, context: str) -> str:
    """Use Groq LLM to answer based on retrieved context."""
    if not context.strip():
        return "Sorry I can't find the details."

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful and friendly Medical Assistant Chatbot. "
                    "Answer the user's medical questions clearly and concisely using ONLY the provided context. "
                    "Provide patient-friendly advice — avoid technical jargon unless necessary. "
                    "If the answer is not in the context, say: "
                    "'I can't tell you, please consult a qualified healthcare professional.'"
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

# -------------------
# Chat Endpoint
# -------------------
@app.post("/chat")
async def chat(request: QueryRequest) -> Dict[str, Any]:
    try:
        query_vector = (await embed_texts([request.question]))[0]
        context = await retrieve_context(query_vector, top_k=5)

        # Debug logs
        print("User query:", request.question)
        print("Retrieved context:", context)

        answer = await get_answer(request.question, context)

        return {
            "original_query": request.question,
            "answer": answer,
            "context_found": bool(context.strip())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"status": "IntelliSnippet API is online"}