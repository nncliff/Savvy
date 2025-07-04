import os
import threading
import time
import logging
from typing import List
from fastapi import FastAPI, Query
from pydantic import BaseModel
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine
import psycopg2
import psycopg2.extensions
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.schema import Document
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)

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

def get_db_connection():
    """Get raw psycopg2 connection for LISTEN/NOTIFY"""
    db_host = os.getenv('POSTGRES_HOST', 'postgres')
    db_port = os.getenv('POSTGRES_PORT', '5432')
    db_name = os.getenv('POSTGRES_DB', 'karakeep')
    db_user = os.getenv('POSTGRES_USER', 'karakeep')
    db_password = os.getenv('POSTGRES_PASSWORD', 'karakeep_password')
    
    return psycopg2.connect(
        host=db_host,
        port=db_port,
        database=db_name,
        user=db_user,
        password=db_password
    )

# In-memory store for bookmarkLinks content
bookmarklinks_data = []
index = None
index_lock = threading.Lock()
indexed_bookmark_ids = set()  # Track which bookmark IDs have been indexed

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

def fetch_new_bookmarklinks():
    """Fetch only new bookmarkLinks that haven't been indexed yet"""
    global indexed_bookmark_ids
    engine = get_engine()
    with engine.connect() as conn:
        if indexed_bookmark_ids:
            # Create a comma-separated string of QUOTED IDs for the NOT IN clause
            indexed_ids_str = ','.join(f"'{id}'" for id in indexed_bookmark_ids)
            query = f'SELECT id, url, title, content, "htmlContent" FROM "bookmarkLinks" WHERE id NOT IN ({indexed_ids_str}) ORDER BY id DESC'
            result = conn.execute(sql_text(query))
        else:
            # If no IDs have been indexed yet, fetch all
            result = conn.execute(sql_text('SELECT id, url, title, content, "htmlContent" FROM "bookmarkLinks" ORDER BY id DESC'))
        
        rows = result.fetchall()
        docs = []
        for row in rows:
            doc_text = f"URL: {row.url}\nTitle: {row.title}\nContent: {row.content or ''}\nHTML: {row.htmlContent or ''}"
            docs.append(Document(text=doc_text, metadata={"id": row.id, "url": row.url}))
        return docs

def update_index():
    """Update the vector index with current bookmark data"""
    global bookmarklinks_data, index, indexed_bookmark_ids
    embed_model = GoogleGenAIEmbedding(model="text-embedding-004")
    
    try:
        new_docs = fetch_new_bookmarklinks()
        
        with index_lock:
            if new_docs:
                # Update the indexed_bookmark_ids set with new document IDs
                new_ids = {doc.metadata["id"] for doc in new_docs}
                logging.info(f"Found {len(new_docs)} new documents to index: {sorted(list(new_ids))}")
                
                indexed_bookmark_ids.update(new_ids)
                
                if index is None:
                    # Create initial index if it doesn't exist
                    index = VectorStoreIndex.from_documents(new_docs, embed_model=embed_model)
                    bookmarklinks_data = new_docs
                    logging.info(f"Initial index created with {len(new_docs)} documents")
                else:
                    # Add new documents to existing index
                    for doc in new_docs:
                        try:
                            index.insert(doc)
                        except Exception as e:
                            logging.error(f"Error inserting document {doc.metadata.get('id', 'unknown')}: {e}")
                            continue
                    
                    bookmarklinks_data.extend(new_docs)
                    logging.info(f"Index updated with {len(new_docs)} new documents. Total indexed IDs: {len(indexed_bookmark_ids)}")
            else:
                logging.info("No new documents to index")
                
    except Exception as e:
        logging.error(f"Error updating index: {e}")
        # If there's an error with incremental update, log it but don't crash
        logging.info("Consider using /rebuild-index endpoint if issues persist")

def validate_index_consistency():
    """Validate that our tracked IDs match what's actually in the database"""
    global indexed_bookmark_ids
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(sql_text('SELECT COUNT(*) as total FROM "bookmarkLinks"'))
            total_bookmarks = result.fetchone().total
            
            logging.info(f"Database has {total_bookmarks} bookmarks, we've indexed {len(indexed_bookmark_ids)}")
            
            if len(indexed_bookmark_ids) > total_bookmarks:
                logging.warning("Indexed more IDs than exist in database - this shouldn't happen!")
                return False
            
            return True
    except Exception as e:
        logging.error(f"Error validating index consistency: {e}")
        return False

def rebuild_index():
    """Rebuild the entire index from scratch (for initialization or recovery)"""
    global bookmarklinks_data, index, indexed_bookmark_ids
    embed_model = GoogleGenAIEmbedding(model="text-embedding-004")
    
    try:
        docs = fetch_bookmarklinks()
        with index_lock:
            bookmarklinks_data = docs
            indexed_bookmark_ids = {doc.metadata["id"] for doc in docs}
            
            if docs:
                index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
                logging.info(f"Index rebuilt with {len(docs)} documents")
            else:
                logging.warning("No documents found for indexing")
    except Exception as e:
        logging.error(f"Error rebuilding index: {e}")

def listen_for_bookmark_changes():
    """Listen for PostgreSQL notifications about bookmark changes"""
    while True:
        try:
            conn = get_db_connection()
            conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            
            with conn.cursor() as cursor:
                # Listen for bookmark notifications
                cursor.execute("LISTEN bookmark_changes;")
                logging.info("Listening for bookmark changes...")
                
                # Initial index build
                rebuild_index()
                
                while True:
                    # Wait for notifications with a timeout
                    if conn.poll() == psycopg2.extensions.POLL_OK:
                        while conn.notifies:
                            notify = conn.notifies.pop(0)
                            logging.info(f"Received notification: {notify.payload}")
                            update_index()
                    time.sleep(1)  # Small delay to prevent busy waiting
                    
        except Exception as e:
            logging.error(f"Error in listener: {e}")
            time.sleep(5)  # Wait before reconnecting
        finally:
            try:
                conn.close()
            except:
                pass

# Start background thread for listening to bookmark changes
threading.Thread(target=listen_for_bookmark_changes, daemon=True).start()

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
            logging.warning("Query attempted but index not ready")
            return {"error": "Index not ready yet. Please try again later."}
        
        try:
            llm = OpenAI(model=DEFAULT_LLM_MODEL)
            query_engine = index.as_query_engine(llm=llm)
            logging.info(f"Processing query: {request.query}")
            response = query_engine.query(request.query)
            logging.info(f"Query processed successfully")
            return {"response": str(response)}
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            return {"error": f"Query processing failed: {str(e)}"}

@app.get("/query")
def query_llamaindex_get(query: str = Query(..., description="Query string")):
    global index
    with index_lock:
        if not index:
            logging.warning("Query attempted but index not ready")
            return {"error": "Index not ready yet. Please try again later."}
        
        try:
            llm = OpenAI(model=DEFAULT_LLM_MODEL)
            query_engine = index.as_query_engine(llm=llm)
            logging.info(f"Processing GET query: {query}")
            response = query_engine.query(query)
            logging.info(f"GET query processed successfully")
            return {"response": str(response)}
        except Exception as e:
            logging.error(f"Error processing GET query: {e}")
            return {"error": f"Query processing failed: {str(e)}"}

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

@app.get("/index-status")
def get_index_status():
    """Get information about the current index status"""
    global index, indexed_bookmark_ids, bookmarklinks_data
    with index_lock:
        consistency_check = validate_index_consistency()
        return {
            "index_ready": index is not None,
            "total_indexed_bookmarks": len(indexed_bookmark_ids),
            "indexed_bookmark_ids": sorted(list(indexed_bookmark_ids)),
            "documents_in_memory": len(bookmarklinks_data),
            "consistency_check_passed": consistency_check
        }

@app.post("/rebuild-index")
def force_rebuild_index():
    """Force a complete rebuild of the index"""
    try:
        rebuild_index()
        return {"message": "Index rebuilt successfully", "status": "success"}
    except Exception as e:
        logging.error(f"Error rebuilding index: {e}")
        return {"error": f"Failed to rebuild index: {str(e)}", "status": "error"}

@app.get("/debug-index")
def debug_index():
    """Debug endpoint to check index state"""
    global index, indexed_bookmark_ids, bookmarklinks_data
    with index_lock:
        return {
            "index_exists": index is not None,
            "index_type": str(type(index)) if index else None,
            "total_indexed_ids": len(indexed_bookmark_ids),
            "total_documents_in_memory": len(bookmarklinks_data),
            "sample_indexed_ids": sorted(list(indexed_bookmark_ids))[:5] if indexed_bookmark_ids else [],
            "sample_documents": [
                {
                    "id": doc.metadata.get("id"),
                    "url": doc.metadata.get("url"),
                    "text_preview": doc.text[:100] + "..." if len(doc.text) > 100 else doc.text
                }
                for doc in bookmarklinks_data[:3]
            ] if bookmarklinks_data else []
        }
