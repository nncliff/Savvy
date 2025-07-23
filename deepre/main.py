from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess

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

@app.get("/")
async def root():
    return {"message": "GraphRAG Query Service", "endpoints": ["/query"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)