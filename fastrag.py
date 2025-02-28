import os
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer, util
from pydantic import BaseModel
from openai import OpenAI
import requests

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

documents = [
    {
        "id": 1,
        "text": "Octopuses have three hearts: two pump blood through the gills, while the third pumps it through the rest of the body."
    },
    {
        "id": 2,
        "text": "The Great Wall of China is not visible from space with the naked eye, contrary to popular belief."
    },
    {
        "id": 3,
        "text": "Honey never spoils - archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible."
    },
    {
        "id": 4,
        "text": "A day on Venus is longer than a year on Venus. It takes 243 Earth days to rotate once on its axis but only 225 Earth days to complete its orbit around the Sun."
    },
    {
        "id": 5,
        "text": "Dog name: Robinho Foguete"
    }
]

model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
# Encode all documents
doc_embeddings = {
    doc["id"]: model.encode(doc["text"], convert_to_tensor=True) for doc in documents
}

class QueryRequest(BaseModel):
    query: str

def find_best_document(query_embedding, documents, doc_embeddings):
    best_doc = {}
    best_score = float("-inf")
    for doc in documents:
        score = util.cos_sim(query_embedding, doc_embeddings[doc["id"]])
        if score > best_score:
            best_score = score
            best_doc = doc
    return best_doc

@app.post("/query")
def query_rag(request: QueryRequest):
    query_embedding = model.encode(request.query, convert_to_tensor=True)
    best_doc = find_best_document(query_embedding, documents, doc_embeddings)
    prompt = f"You are an AI assistant, you have to inform a document use for response. Answer based on this document: {best_doc.get('text')} \n\n User: {request.query} \n Assistant:"

    try:
        res = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt}
            ],
        )
        return {"response": res.choices[0].message.content}
    except openai_client.error.OpenAIError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/queryLocal")
def query_rag_local(request: QueryRequest):
    query_embedding = model.encode(request.query, convert_to_tensor=True)
    best_doc = find_best_document(query_embedding, documents, doc_embeddings)
    prompt = f"You are an AI assistant, you have to inform a document use for response. Answer based on this document: {best_doc.get('text')} \n\n User: {request.query} \n Assistant:"

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False
            }
        )
        if response.status_code == 200:
            result = response.json()
            return {"response": result["response"]}
        else:
            raise HTTPException(status_code=response.status_code, detail="Error from Ollama API")
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))