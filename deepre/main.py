from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import random

app = FastAPI(title="GraphRAG Query Service")

class QueryRequest(BaseModel):
    query: str = "give me 2 topics related to reinforcement learning"
    method: str = "global"
    root: str = "./"

class QueryResponse(BaseModel):
    result: str
    query: str
    method: str

@app.post("/query", response_model=QueryResponse)
async def graphrag_query(request: QueryRequest):
    try:
        result = subprocess.run([
            "graphrag", "query",
            "--root", request.root,
            "--method", request.method,
            "--query", request.query
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"GraphRAG failed: {result.stderr}")
        
        return QueryResponse(result=result.stdout, query=request.query, method=request.method)
    
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Query timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/random-query", response_model=QueryResponse)
async def random_graphrag_query(request: QueryRequest):
    # Predefined queries for random selection
    random_queries = [
        "Give me 2 topics from the index, forcing unexpected connections between distant fields",
        "Give me 2 topics from the index, looking for structural similarities across scales",
        "Give me 2 topics from the index, applying constraints from one domain to another",
        "Give me 2 topics from the index, seeking emergent properties from combinations",
        "Give me 2 topics from the index, using metaphor as a discovery tool",
        "Give me 2 topics from the index, questioning fundamental assumptions through cross-domain lenses",
        "Give me 2 topics from the index, gain new insights through the collision of these ideas"
    ]
    
    # Randomly select a query
    selected_query = random.choice(random_queries)
    
    try:
        result = subprocess.run([
            "graphrag", "query",
            "--root", request.root,
            "--method", request.method,
            "--query", selected_query
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"GraphRAG failed: {result.stderr}")
        
        return QueryResponse(result=result.stdout, query=selected_query, method=request.method)
    
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Query timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "GraphRAG Query Service", "endpoints": ["/query", "/random-query"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)