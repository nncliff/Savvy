import os
import threading
import time
from typing import List
from fastapi import FastAPI, Query
from pydantic import BaseModel
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.schema import Document
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Database connection setup
def get_engine() -> Engine:
    db_url = os.getenv(
        "DATABASE_URL",
        f"postgresql://{os.getenv('POSTGRES_USER','karakeep')}:{os.getenv('POSTGRES_PASSWORD','karakeep_password')}@{os.getenv('POSTGRES_HOST','postgres')}:{os.getenv('POSTGRES_PORT','5432')}/{os.getenv('POSTGRES_DB','karakeep')}"
    )
    # Fix: SQLAlchemy expects 'postgresql://' not 'postgres://'
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    return create_engine(db_url)

# In-memory store for bookmarkLinks content
bookmarklinks_data = []
index = None
index_lock = threading.Lock()

# Fetch bookmarkLinks from DB
def fetch_bookmarklinks():
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(sql_text('SELECT id, url, title, content, "htmlContent" FROM "bookmarkLinks" ORDER BY id DESC'))
        rows = result.fetchall()
        docs = []
        for row in rows:
            doc_text = f"URL: {row.url}\nTitle: {row.title}\nContent: {row.content or ''}\nHTML: {row.htmlContent or ''}"
            docs.append(Document(text=doc_text, metadata={"id": row.id, "url": row.url}))
        return docs

def update_index_periodically():
    global bookmarklinks_data, index
    while True:
        docs = fetch_bookmarklinks()
        with index_lock:
            bookmarklinks_data = docs
            if docs:
                index = VectorStoreIndex.from_documents(docs)
        time.sleep(300)  # 5 minutes

# Start background thread for periodic update
threading.Thread(target=update_index_periodically, daemon=True).start()

@app.get("/")
def read_root():
    return {"message": "LlamaIndex RAG API is running"}

class QueryRequest(BaseModel):
    query: str

# Set default LLM model name
DEFAULT_LLM_MODEL = "gpt-4o-mini"

@app.post("/query")
def query_llamaindex(request: QueryRequest):
    global index
    with index_lock:
        if not index:
            return {"error": "Index not ready yet. Please try again later."}
        llm = OpenAI(model=DEFAULT_LLM_MODEL)
        query_engine = index.as_query_engine(llm=llm)
        response = query_engine.query(request.query)
        return {"response": str(response)}

@app.get("/query")
def query_llamaindex_get(query: str = Query(..., description="Query string")):
    global index
    with index_lock:
        if not index:
            return {"error": "Index not ready yet. Please try again later."}
        llm = OpenAI(model=DEFAULT_LLM_MODEL)
        query_engine = index.as_query_engine(llm=llm)
        response = query_engine.query(query)
        return {"response": str(response)}

@app.get("/top3-bookmarklinks")
def get_top3_bookmarklinks():
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(sql_text('SELECT id, url, title, content, "htmlContent" FROM "bookmarkLinks" ORDER BY id DESC LIMIT 3'))
        rows = result.fetchall()
        return [
            {
                "id": row.id,
                "url": row.url,
                "title": row.title,
                "content": row.content,
                "htmlContent": row.htmlContent
            }
            for row in rows
        ]
