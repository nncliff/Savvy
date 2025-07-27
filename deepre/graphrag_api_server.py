#!/usr/bin/env python3
"""
GraphRAG API Server
Provides REST API endpoints for GraphRAG queries
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
from typing import Any, Dict, Optional
import json
from datetime import datetime

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("FastAPI not installed. Install with: pip install fastapi uvicorn")
    sys.exit(1)

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from run_graphrag_query import GraphRAGQueryEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="GraphRAG Query API",
    description="REST API for querying GraphRAG knowledge base",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global query engine
query_engine: Optional[GraphRAGQueryEngine] = None


class QueryRequest(BaseModel):
    query: str
    search_type: str = "local"
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    clean_response: bool = False


class QueryResponse(BaseModel):
    query: str
    search_type: str
    result: Any
    status: str
    timestamp: str
    error: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the GraphRAG query engine on startup."""
    global query_engine
    try:
        query_engine = GraphRAGQueryEngine()
        logger.info("GraphRAG API server started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize query engine: {str(e)}")
        raise


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "GraphRAG Query API",
        "version": "1.0.0",
        "endpoints": {
            "local_search": "/search/local",
            "global_search": "/search/global",
            "query": "/query",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "engine_ready": query_engine is not None
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    General query endpoint that supports both local and global search.
    """
    if query_engine is None:
        raise HTTPException(status_code=500, detail="Query engine not initialized")
    
    try:
        # Prepare search parameters
        search_params = {}
        if request.max_tokens:
            search_params["max_tokens"] = request.max_tokens
        if request.temperature:
            search_params["temperature"] = request.temperature
        
        # Perform search
        result = query_engine.search(
            query=request.query,
            search_type=request.search_type,
            clean_response=request.clean_response,
            **search_params
        )
        
        # Add timestamp
        result["timestamp"] = datetime.now().isoformat()
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/local")
async def local_search_endpoint(request: QueryRequest):
    """Local search endpoint."""
    request.search_type = "local"
    return await query_endpoint(request)


@app.post("/search/global")
async def global_search_endpoint(request: QueryRequest):
    """Global search endpoint."""
    request.search_type = "global"
    return await query_endpoint(request)


@app.get("/ui", response_class=HTMLResponse)
async def web_interface():
    """Simple web interface for testing queries."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>GraphRAG Query Interface</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { margin: 20px 0; }
            input, select, textarea, button { padding: 10px; margin: 5px; border: 1px solid #ddd; border-radius: 4px; }
            #query { width: 100%; min-height: 100px; }
            #result { width: 100%; min-height: 300px; background: #f5f5f5; font-family: monospace; }
            button { background: #007bff; color: white; cursor: pointer; }
            button:hover { background: #0056b3; }
            .loading { color: #666; }
        </style>
    </head>
    <body>
        <h1>GraphRAG Query Interface</h1>
        
        <div class="container">
            <label for="query">Query:</label><br>
            <textarea id="query" placeholder="Enter your search query here..."></textarea>
        </div>
        
        <div class="container">
            <label for="searchType">Search Type:</label>
            <select id="searchType">
                <option value="local">Local Search</option>
                <option value="global">Global Search</option>
            </select>
            
            <label for="maxTokens">Max Tokens (optional):</label>
            <input type="number" id="maxTokens" placeholder="2000">
            
            <label>
                <input type="checkbox" id="cleanResponse"> Clean response (final answer only)
            </label>
            
            <button onclick="submitQuery()">Search</button>
        </div>
        
        <div class="container">
            <label for="result">Result:</label><br>
            <textarea id="result" readonly></textarea>
        </div>
        
        <script>
            async function submitQuery() {
                const query = document.getElementById('query').value;
                const searchType = document.getElementById('searchType').value;
                const maxTokens = document.getElementById('maxTokens').value;
                const cleanResponse = document.getElementById('cleanResponse').checked;
                const resultArea = document.getElementById('result');
                
                if (!query.trim()) {
                    alert('Please enter a query');
                    return;
                }
                
                resultArea.value = 'Searching...';
                
                try {
                    const payload = {
                        query: query,
                        search_type: searchType,
                        clean_response: cleanResponse
                    };
                    
                    if (maxTokens) {
                        payload.max_tokens = parseInt(maxTokens);
                    }
                    
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(payload)
                    });
                    
                    const data = await response.json();
                    
                    if (cleanResponse && data.status === 'success') {
                        // Show just the clean result
                        resultArea.value = data.result;
                    } else {
                        // Show full JSON response
                        resultArea.value = JSON.stringify(data, null, 2);
                    }
                    
                } catch (error) {
                    resultArea.value = 'Error: ' + error.message;
                }
            }
            
            // Allow Enter to submit (Ctrl+Enter for new line)
            document.getElementById('query').addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && e.ctrlKey) {
                    submitQuery();
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


def main():
    """Run the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GraphRAG API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    logger.info(f"Starting GraphRAG API server on {args.host}:{args.port}")
    logger.info(f"Web interface available at: http://{args.host}:{args.port}/ui")
    logger.info(f"API documentation at: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "graphrag_api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()