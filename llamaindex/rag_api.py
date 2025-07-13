import os
import sys
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
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.schema import Document
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from dotenv import load_dotenv
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

load_dotenv()

app = FastAPI()

# Set default LLM model name
DEFAULT_LLM_MODEL = "gpt-4o-mini"

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

# Storage configuration
PERSIST_DIR = os.getenv("INDEX_PERSIST_DIR", "./storage")

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

def load_existing_index():
    """Load existing index from storage if it exists"""
    global index, indexed_bookmark_ids, bookmarklinks_data
    embed_model = GoogleGenAIEmbedding(model="text-embedding-004")
    
    try:
        # Check if storage directory exists and has index files
        if os.path.exists(PERSIST_DIR) and os.path.exists(os.path.join(PERSIST_DIR, "docstore.json")):
            # Load existing index
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context, embed_model=embed_model)
            
            # Rebuild indexed_bookmark_ids from existing documents
            docs = fetch_bookmarklinks()
            indexed_bookmark_ids = {doc.metadata["id"] for doc in docs}
            bookmarklinks_data = docs
            
            logging.info(f"Loaded existing index with {len(indexed_bookmark_ids)} indexed bookmarks")
        else:
            logging.info("No existing index found, will create new one when documents are available")
    except Exception as e:
        logging.error(f"Error loading existing index: {e}")
        logging.info("Will create fresh index")

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
                    # Ensure storage directory exists
                    os.makedirs(PERSIST_DIR, exist_ok=True)
                    
                    # Create initial index with fresh storage context
                    storage_context = StorageContext.from_defaults()
                    index = VectorStoreIndex.from_documents(new_docs, storage_context=storage_context, embed_model=embed_model)
                    
                    # Persist the created index
                    index.storage_context.persist(persist_dir=PERSIST_DIR)
                    bookmarklinks_data = new_docs
                    logging.info(f"Initial index created with {len(new_docs)} documents and persisted to {PERSIST_DIR}")
                else:
                    # Add new documents to existing index
                    for doc in new_docs:
                        try:
                            index.insert(doc)
                        except Exception as e:
                            logging.error(f"Error inserting document {doc.metadata.get('id', 'unknown')}: {e}")
                            continue
                    
                    # Persist the updated index
                    index.storage_context.persist(persist_dir=PERSIST_DIR)
                    bookmarklinks_data.extend(new_docs)
                    logging.info(f"Index updated with {len(new_docs)} new documents and persisted. Total indexed IDs: {len(indexed_bookmark_ids)}")
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
                # Ensure storage directory exists
                os.makedirs(PERSIST_DIR, exist_ok=True)
                
                # Process documents in batches to avoid rate limiting
                batch_size = 20
                
                logging.info(f"Processing {len(docs)} documents in batches of {batch_size}")
                
                # Create initial index with first batch
                first_batch = docs[:batch_size]
                storage_context = StorageContext.from_defaults()
                index = VectorStoreIndex.from_documents(first_batch, storage_context=storage_context, embed_model=embed_model)
                logging.info(f"Created initial index with {len(first_batch)} documents")
                
                # Process remaining documents in batches
                for i in range(batch_size, len(docs), batch_size):
                    batch = docs[i:i+batch_size]
                    
                    # Wait before processing next batch
                    if i > batch_size:  # Don't wait before the first additional batch
                        logging.info(f"Waiting 60 seconds before processing batch {i//batch_size + 1}...")
                        time.sleep(60)
                    
                    # Add documents one by one to existing index
                    for doc in batch:
                        try:
                            index.insert(doc)
                        except Exception as e:
                            logging.error(f"Error inserting document {doc.metadata.get('id', 'unknown')}: {e}")
                            continue
                    
                    logging.info(f"Processed batch {i//batch_size + 1} with {len(batch)} documents")
                
                # Persist the final index
                index.storage_context.persist(persist_dir=PERSIST_DIR)
                logging.info(f"Index rebuilt with {len(docs)} documents and persisted to {PERSIST_DIR}")
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
                
                # Load existing index on startup, then listen for changes
                load_existing_index()
                
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

# Deep Research Agent Implementation
class DeepResearchAgent:
    def __init__(self):
        self.llm = OpenAI(model=DEFAULT_LLM_MODEL)
        self.search_tool = None
        self._setup_search_tool()
    
    def _extract_json_from_response(self, response_text: str) -> dict:
        """Extract JSON from LLM response with fallback handling"""
        try:
            response_text = response_text.strip()
            
            # Remove common prefixes that might interfere
            prefixes_to_remove = [
                "```json\n", "```\n", "Here is the JSON:", "JSON:", 
                "Response:", "Here's the response:", "The answer is:"
            ]
            for prefix in prefixes_to_remove:
                if response_text.startswith(prefix):
                    response_text = response_text[len(prefix):].strip()
            
            # Remove common suffixes
            suffixes_to_remove = ["\n```", "```"]
            for suffix in suffixes_to_remove:
                if response_text.endswith(suffix):
                    response_text = response_text[:-len(suffix)].strip()
            
            # Find JSON boundaries
            if '{' in response_text and '}' in response_text:
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                json_text = response_text[start:end]
                return json.loads(json_text)
            else:
                return None
                
        except (json.JSONDecodeError, Exception):
            return None
    
    def _setup_search_tool(self):
        """Setup the local document search tool using the existing index"""
        global index
        with index_lock:
            if index is not None:
                query_engine = index.as_query_engine(llm=self.llm)
                self.search_tool = QueryEngineTool(
                    query_engine=query_engine,
                    metadata=ToolMetadata(
                        name="local_document_search",
                        description="Search through the local bookmark database for relevant information. "
                                  "Use this tool to find documents, articles, and content related to your research query."
                    )
                )
    
    def _create_plan_prompt(self, research_query: str) -> str:
        return f"""
PHASE 1: PLAN
You are conducting deep research on: "{research_query}"

Break down this research topic into 3-5 specific questions or areas that need investigation.
Each question should be:
- Specific and focused
- Searchable in a document database
- Contributing to understanding the overall topic

IMPORTANT: You must respond with ONLY valid JSON, no other text before or after.

{{
    "research_topic": "{research_query}",
    "plan": [
        "Question 1: [specific question]",
        "Question 2: [specific question]",
        "Question 3: [specific question]"
    ]
}}
"""

    def _create_think_prompt(self, plan: dict, question: str) -> str:
        return f"""
PHASE 2: THINK
Research Topic: {plan['research_topic']}
Current Question: {question}

Before searching, think about:
1. What specific keywords or concepts should I search for?
2. What type of documents would likely contain this information?
3. What related terms or synonyms might be relevant?

Create 2-3 search queries that would help answer this question.

IMPORTANT: You must respond with ONLY valid JSON, no other text before or after.

{{
    "question": "{question}",
    "reasoning": "Your reasoning about what to search for",
    "search_queries": [
        "search query 1",
        "search query 2", 
        "search query 3"
    ]
}}
"""

    def _create_analyze_prompt(self, research_query: str, all_findings: list) -> str:
        findings_text = "\n\n".join([
            f"Question: {finding['question']}\nFindings: {finding['search_results']}"
            for finding in all_findings
        ])
        
        return f"""
PHASE 4: ANALYZE
Research Topic: {research_query}

Based on all the findings from your search actions, provide a comprehensive analysis:

FINDINGS:
{findings_text}

Provide a structured analysis including:
1. Key insights discovered
2. Patterns or themes identified
3. Gaps in available information
4. Conclusions and recommendations
5. Areas for further research

Focus on synthesizing the information rather than just summarizing individual findings.
"""

    def conduct_deep_research(self, research_query: str) -> dict:
        """Conduct deep research using Plan → Think → Action → Analyze pattern"""
        
        if self.search_tool is None:
            self._setup_search_tool()
            if self.search_tool is None:
                return {"error": "Search tool not available. Index may not be ready."}
        
        try:
            result = {
                "research_query": research_query,
                "process_steps": [],
                "plan": None,
                "findings": [],
                "analysis": None
            }
            
            # PHASE 1: PLAN
            print(f"🔍 DEEP RESEARCH STARTED: {research_query}", flush=True)
            print("="*80, flush=True)
            print("📋 PHASE 1: PLANNING", flush=True)
            print(f"Topic: {research_query}", flush=True)
            logging.info(f"🔍 DEEP RESEARCH STARTED: {research_query}")
            logging.info("="*80)
            logging.info("📋 PHASE 1: PLANNING")
            logging.info(f"Topic: {research_query}")
            sys.stdout.flush()
            
            plan_prompt = self._create_plan_prompt(research_query)
            print("Sending planning prompt to LLM...", flush=True)
            logging.info("Sending planning prompt to LLM...")
            
            result["process_steps"].append({
                "phase": "1_PLAN",
                "description": f"Planning research approach for: {research_query}",
                "prompt": plan_prompt,
                "status": "executing"
            })
            
            plan_response = self.llm.complete(plan_prompt)
            
            # Parse plan response with error handling
            plan_text = str(plan_response).strip()
            logging.info(f"Plan response received: {plan_text[:200]}...")
            
            plan = self._extract_json_from_response(plan_text)
            if plan is None:
                # Fallback plan
                plan = {
                    "research_topic": research_query,
                    "plan": [
                        f"Question 1: What are the key aspects of {research_query}?",
                        f"Question 2: What are the current developments in {research_query}?",
                        f"Question 3: What are the implications and applications of {research_query}?"
                    ]
                }
                logging.warning("Using fallback plan due to invalid JSON response")
            
            result["plan"] = plan
            
            print("✅ Planning completed!", flush=True)
            print(f"Generated {len(plan['plan'])} research questions:", flush=True)
            logging.info("✅ Planning completed!")
            logging.info(f"Generated {len(plan['plan'])} research questions:")
            for i, question in enumerate(plan['plan'], 1):
                print(f"   {i}. {question}", flush=True)
                logging.info(f"   {i}. {question}")
            
            result["process_steps"][-1].update({
                "response": str(plan_response),
                "parsed_result": plan,
                "status": "completed"
            })
            
            # PHASE 2 & 3: THINK → ACTION for each question
            all_findings = []
            question_number = 1
            
            for question in plan['plan']:
                logging.info("="*80)
                logging.info(f"🤔 PHASE 2: THINKING (Question {question_number})")
                logging.info(f"Question: {question}")
                
                # PHASE 2: THINK
                think_prompt = self._create_think_prompt(plan, question)
                logging.info("Generating search strategy...")
                
                result["process_steps"].append({
                    "phase": f"2_THINK_Q{question_number}",
                    "description": f"Thinking about search strategy for: {question}",
                    "question": question,
                    "prompt": think_prompt,
                    "status": "executing"
                })
                
                think_response = self.llm.complete(think_prompt)
                
                # Parse think response with error handling
                think_text = str(think_response).strip()
                logging.info(f"Think response received: {think_text[:200]}...")
                
                think_data = self._extract_json_from_response(think_text)
                if think_data is None:
                    # Fallback think data
                    think_data = {
                        "question": question,
                        "reasoning": f"Search for information related to {question}",
                        "search_queries": [
                            question.replace("Question ", "").replace(":", ""),
                            f"{research_query} {question.split(':')[-1] if ':' in question else question}",
                            f"information about {question.split(':')[-1] if ':' in question else question}"
                        ]
                    }
                    logging.warning("Using fallback think data due to invalid JSON response")
                
                logging.info("✅ Strategy generated!")
                logging.info(f"Reasoning: {think_data['reasoning']}")
                logging.info(f"Search queries: {think_data['search_queries']}")
                
                result["process_steps"][-1].update({
                    "response": str(think_response),
                    "parsed_result": think_data,
                    "status": "completed"
                })
                
                # PHASE 3: ACTION - Search for each query
                logging.info(f"🔎 PHASE 3: ACTION (Question {question_number})")
                search_results = []
                query_number = 1
                
                for search_query in think_data['search_queries']:
                    logging.info(f"   Executing search {query_number}: '{search_query}'")
                    
                    result["process_steps"].append({
                        "phase": f"3_ACTION_Q{question_number}_S{query_number}",
                        "description": f"Searching for: {search_query}",
                        "search_query": search_query,
                        "status": "executing"
                    })
                    
                    try:
                        search_result = self.search_tool.call(search_query)
                        search_results.append({
                            "query": search_query,
                            "result": str(search_result)
                        })
                        
                        # Log result preview
                        result_preview = str(search_result)[:200] + "..." if len(str(search_result)) > 200 else str(search_result)
                        logging.info(f"   ✅ Search {query_number} completed: {result_preview}")
                        
                        result["process_steps"][-1].update({
                            "result": str(search_result),
                            "status": "completed"
                        })
                        
                    except Exception as e:
                        logging.error(f"   ❌ Search {query_number} failed: {e}")
                        error_msg = f"Search failed: {str(e)}"
                        search_results.append({
                            "query": search_query,
                            "result": error_msg
                        })
                        
                        result["process_steps"][-1].update({
                            "result": error_msg,
                            "status": "failed",
                            "error": str(e)
                        })
                    
                    query_number += 1
                
                all_findings.append({
                    "question": question,
                    "reasoning": think_data['reasoning'],
                    "search_queries": think_data['search_queries'],
                    "search_results": search_results
                })
                
                question_number += 1
            
            result["findings"] = all_findings
            
            # PHASE 4: ANALYZE
            logging.info("="*80)
            logging.info("📊 PHASE 4: ANALYZING")
            logging.info("Synthesizing all findings into comprehensive insights...")
            
            analyze_prompt = self._create_analyze_prompt(research_query, all_findings)
            
            result["process_steps"].append({
                "phase": "4_ANALYZE",
                "description": "Analyzing all findings and synthesizing insights",
                "prompt": analyze_prompt,
                "status": "executing"
            })
            
            analysis = self.llm.complete(analyze_prompt)
            result["analysis"] = str(analysis)
            
            logging.info("✅ Analysis completed!")
            analysis_preview = str(analysis)[:300] + "..." if len(str(analysis)) > 300 else str(analysis)
            logging.info(f"Analysis preview: {analysis_preview}")
            
            result["process_steps"][-1].update({
                "response": str(analysis),
                "status": "completed"
            })
            
            # Add summary
            result["summary"] = {
                "total_steps": len(result["process_steps"]),
                "completed_steps": len([s for s in result["process_steps"] if s["status"] == "completed"]),
                "failed_steps": len([s for s in result["process_steps"] if s["status"] == "failed"]),
                "questions_researched": len(plan['plan']),
                "total_searches_performed": sum(len(f['search_queries']) for f in all_findings)
            }
            
            # Final summary log
            logging.info("="*80)
            logging.info("🏁 DEEP RESEARCH COMPLETED!")
            logging.info(f"📈 Summary:")
            logging.info(f"   • Questions researched: {result['summary']['questions_researched']}")
            logging.info(f"   • Total searches: {result['summary']['total_searches_performed']}")
            logging.info(f"   • Completed steps: {result['summary']['completed_steps']}")
            logging.info(f"   • Failed steps: {result['summary']['failed_steps']}")
            logging.info("="*80)
            
            return result
            
        except Exception as e:
            logging.error(f"Deep research error: {e}")
            return {"error": f"Deep research failed: {str(e)}"}

# Global deep research agent
deep_research_agent = DeepResearchAgent()

# Import CrewAI system
try:
    from crewai_deep_research import crewai_research_system
    CREWAI_AVAILABLE = True
    logging.info("CrewAI multi-agent system loaded successfully")
except ImportError as e:
    CREWAI_AVAILABLE = False
    logging.warning(f"CrewAI system not available: {e}")
    crewai_research_system = None

@app.get("/")
def read_root():
    print("🔥 ROOT ENDPOINT CALLED - TESTING LOG VISIBILITY", flush=True)
    logging.info("🔥 ROOT ENDPOINT CALLED - TESTING LOG VISIBILITY")
    return {"message": "LlamaIndex RAG API is running"}

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_llamaindex(request: QueryRequest):
    """Process query using CrewAI multi-agent system with advanced reasoning (GPT-4o + o1-preview)"""
    if not CREWAI_AVAILABLE:
        return {"error": "CrewAI multi-agent system not available. Please install crewai and crewai-tools packages."}
    
    if not request.query.strip():
        return {"error": "Query cannot be empty"}
    
    try:
        print(f"🚀 CREWAI QUERY PROCESSING STARTED: {request.query}", flush=True)
        print("Using enhanced multi-agent system with GPT-4o + o1-preview reasoning", flush=True)
        logging.info(f"🚀 CREWAI QUERY PROCESSING STARTED: {request.query}")
        logging.info("Using enhanced multi-agent system with advanced reasoning")
        
        result = crewai_research_system.conduct_research(request.query)
        
        # Format response for backward compatibility while providing CrewAI results
        if "error" in result:
            return result
        
        # Extract the final report as the main response for compatibility
        main_response = result.get("final_report", "No research report available")
        
        return {
            "response": main_response,
            "metadata": {
                "query": request.query,
                "model": "CrewAI Multi-Agent (GPT-4o + o1-preview)",
                "method": "crewai_multi_agent_research",
                "agent_count": 8,
                "reasoning_model": "o1-preview"
            },
            "system_info": {
                "research_method": "CrewAI Multi-Agent System",
                "agents_used": [
                    "Hybrid Research Developer (SQLite + RAG + Multi-language)",
                    "Research Supervisor (data-driven question generation)",
                    "Elite Research Critic (≥8.5 novelty + o1-preview reasoning)",
                    "Strategic Analyzer (methodology design)", 
                    "Research Student (LlamaIndex search)",
                    "Research Philosopher (strategic guidance)",
                    "Deep Thinker (o1-preview synthesis)",
                    "Research Reporter (final reporting)"
                ],
                "capabilities": [
                    "Multi-language support (English/Chinese/Japanese)",
                    "Hybrid analysis (SQLite + LlamaIndex RAG)",
                    "Advanced reasoning with o1-preview",
                    "≥8.5/10 novelty requirements",
                    "Cross-cultural research opportunities"
                ]
            },
            "crewai_details": {
                "agent_outputs": result.get("agent_outputs", {}),
                "metadata": result.get("metadata", {})
            }
        }
    except Exception as e:
        logging.error(f"Error processing query with CrewAI: {e}")
        return {"error": f"Failed to process query with CrewAI: {str(e)}"}

@app.get("/query")
def query_llamaindex_get(query: str = Query(..., description="Query string")):
    """Process GET query using CrewAI multi-agent system with advanced reasoning (GPT-4o + o1-preview)"""
    if not CREWAI_AVAILABLE:
        return {"error": "CrewAI multi-agent system not available. Please install crewai and crewai-tools packages."}
    
    if not query.strip():
        return {"error": "Query cannot be empty"}
    
    try:
        print(f"🚀 CREWAI GET QUERY PROCESSING STARTED: {query}", flush=True)
        print("Using enhanced multi-agent system with GPT-4o + o1-preview reasoning", flush=True)
        logging.info(f"🚀 CREWAI GET QUERY PROCESSING STARTED: {query}")
        logging.info("Using enhanced multi-agent system with advanced reasoning")
        
        result = crewai_research_system.conduct_research(query)
        
        # Format response for backward compatibility while providing CrewAI results
        if "error" in result:
            return result
        
        # Extract the final report as the main response for compatibility
        main_response = result.get("final_report", "No research report available")
        
        return {
            "response": main_response,
            "metadata": {
                "query": query,
                "model": "CrewAI Multi-Agent (GPT-4o + o1-preview)",
                "method": "crewai_multi_agent_research",
                "agent_count": 8,
                "reasoning_model": "o1-preview"
            },
            "system_info": {
                "research_method": "CrewAI Multi-Agent System",
                "agents_used": [
                    "Hybrid Research Developer (SQLite + RAG + Multi-language)",
                    "Research Supervisor (data-driven question generation)",
                    "Elite Research Critic (≥8.5 novelty + o1-preview reasoning)",
                    "Strategic Analyzer (methodology design)", 
                    "Research Student (LlamaIndex search)",
                    "Research Philosopher (strategic guidance)",
                    "Deep Thinker (o1-preview synthesis)",
                    "Research Reporter (final reporting)"
                ],
                "capabilities": [
                    "Multi-language support (English/Chinese/Japanese)",
                    "Hybrid analysis (SQLite + LlamaIndex RAG)",
                    "Advanced reasoning with o1-preview",
                    "≥8.5/10 novelty requirements",
                    "Cross-cultural research opportunities"
                ]
            },
            "crewai_details": {
                "agent_outputs": result.get("agent_outputs", {}),
                "metadata": result.get("metadata", {})
            }
        }
    except Exception as e:
        logging.error(f"Error processing GET query with CrewAI: {e}")
        return {"error": f"Failed to process query with CrewAI: {str(e)}"}

@app.post("/simple-query")
def simple_query_llamaindex(request: QueryRequest):
    """Simple query without deep research (legacy behavior)"""
    global index
    with index_lock:
        if not index:
            logging.warning("Simple query attempted but index not ready")
            return {"error": "Index not ready yet. Please try again later."}
        
        try:
            llm = OpenAI(model=DEFAULT_LLM_MODEL)
            query_engine = index.as_query_engine(llm=llm)
            logging.info(f"Processing simple query: {request.query}")
            response = query_engine.query(request.query)
            logging.info(f"Simple query processed successfully")
            return {
                "response": str(response),
                "metadata": {
                    "query": request.query,
                    "model": DEFAULT_LLM_MODEL,
                    "method": "simple"
                }
            }
        except Exception as e:
            logging.error(f"Error processing simple query: {e}")
            return {"error": f"Simple query processing failed: {str(e)}"}

@app.get("/simple-query")
def simple_query_llamaindex_get(query: str = Query(..., description="Simple query string")):
    """Simple GET query without deep research (legacy behavior)"""
    global index
    with index_lock:
        if not index:
            logging.warning("Simple GET query attempted but index not ready")
            return {"error": "Index not ready yet. Please try again later."}
        
        try:
            llm = OpenAI(model=DEFAULT_LLM_MODEL)
            query_engine = index.as_query_engine(llm=llm)
            logging.info(f"Processing simple GET query: {query}")
            response = query_engine.query(query)
            logging.info(f"Simple GET query processed successfully")
            return {
                "response": str(response),
                "metadata": {
                    "query": query,
                    "model": DEFAULT_LLM_MODEL,
                    "method": "simple"
                }
            }
        except Exception as e:
            logging.error(f"Error processing simple GET query: {e}")
            return {"error": f"Simple query processing failed: {str(e)}"}

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

class DeepResearchRequest(BaseModel):
    research_query: str

@app.post("/deep-research")
def conduct_deep_research_endpoint(request: DeepResearchRequest):
    """Legacy single-agent deep research (Plan → Think → Action → Analyze pattern). Use /query for enhanced CrewAI multi-agent research."""
    global deep_research_agent
    
    if not request.research_query.strip():
        return {"error": "Research query cannot be empty"}
    
    try:
        print(f"⚠️ Using legacy deep research endpoint. Consider using /query for enhanced CrewAI multi-agent research.", flush=True)
        logging.info(f"Legacy deep research endpoint used for: {request.research_query}")
        
        # Refresh the search tool if index was updated
        deep_research_agent._setup_search_tool()
        
        result = deep_research_agent.conduct_deep_research(request.research_query)
        
        # Add note about enhanced endpoint
        if isinstance(result, dict):
            result["note"] = "This is legacy single-agent research. Use /query endpoint for enhanced CrewAI multi-agent research with GPT-4o + o1-preview."
        
        return result
    except Exception as e:
        logging.error(f"Deep research endpoint error: {e}")
        return {"error": f"Deep research failed: {str(e)}"}

@app.get("/deep-research")
def conduct_deep_research_get(research_query: str = Query(..., description="Research query for deep analysis")):
    """GET endpoint for deep research"""
    global deep_research_agent
    
    if not research_query.strip():
        return {"error": "Research query cannot be empty"}
    
    try:
        # Refresh the search tool if index was updated
        deep_research_agent._setup_search_tool()
        
        result = deep_research_agent.conduct_deep_research(research_query)
        return result
    except Exception as e:
        logging.error(f"Deep research GET endpoint error: {e}")
        return {"error": f"Deep research failed: {str(e)}"}

# CrewAI Multi-Agent Research Endpoints

class CrewAIResearchRequest(BaseModel):
    research_query: str

@app.post("/crewai-research")
def conduct_crewai_research(request: CrewAIResearchRequest):
    """Conduct research using CrewAI multi-agent system with 8 specialized agents"""
    if not CREWAI_AVAILABLE:
        return {"error": "CrewAI multi-agent system not available. Please install crewai and crewai-tools packages."}
    
    if not request.research_query.strip():
        return {"error": "Research query cannot be empty"}
    
    try:
        logging.info(f"Starting CrewAI multi-agent research for: {request.research_query}")
        
        result = crewai_research_system.conduct_research(request.research_query)
        
        # Add system information
        result["system_info"] = {
            "research_method": "CrewAI Multi-Agent System",
            "agents_used": [
                "Research Supervisor (data-driven question generation)",
                "Strategic Analyzer (methodology design)", 
                "Research Student (LlamaIndex search)",
                "Research Developer (computational analysis)",
                "Research Philosopher (strategic guidance)",
                "Deep Thinker (synthesis)",
                "Professional Critic (quality control)",
                "Research Reporter (final reporting)"
            ],
            "agent_count": 8,
            "workflow": "Flexible sequential with context sharing"
        }
        
        return result
        
    except Exception as e:
        logging.error(f"CrewAI research error: {e}")
        return {"error": f"CrewAI research failed: {str(e)}"}

@app.get("/crewai-research")
def conduct_crewai_research_get(research_query: str = Query(..., description="Research query for CrewAI multi-agent analysis")):
    """GET endpoint for CrewAI multi-agent research"""
    if not CREWAI_AVAILABLE:
        return {"error": "CrewAI multi-agent system not available. Please install crewai and crewai-tools packages."}
    
    if not research_query.strip():
        return {"error": "Research query cannot be empty"}
    
    try:
        logging.info(f"Starting CrewAI multi-agent research (GET) for: {research_query}")
        
        result = crewai_research_system.conduct_research(research_query)
        
        # Add system information
        result["system_info"] = {
            "research_method": "CrewAI Multi-Agent System",
            "agents_used": [
                "Research Supervisor (data-driven question generation)",
                "Strategic Analyzer (methodology design)", 
                "Research Student (LlamaIndex search)",
                "Research Developer (computational analysis)",
                "Research Philosopher (strategic guidance)",
                "Deep Thinker (synthesis)",
                "Professional Critic (quality control)",
                "Research Reporter (final reporting)"
            ],
            "agent_count": 8,
            "workflow": "Flexible sequential with context sharing"
        }
        
        return result
        
    except Exception as e:
        logging.error(f"CrewAI research GET error: {e}")
        return {"error": f"CrewAI research failed: {str(e)}"}

@app.get("/research-methods")
def get_available_research_methods():
    """Get information about available research methods"""
    methods = {
        "simple_query": {
            "endpoint": "/simple-query",
            "description": "Basic LlamaIndex search without deep analysis",
            "features": ["Single query", "Fast response", "Vector similarity search"],
            "use_case": "Quick information lookup"
        },
        "deep_research": {
            "endpoint": "/deep-research", 
            "description": "Advanced single-agent deep research with Plan→Think→Action→Analyze pattern",
            "features": ["Multi-phase analysis", "Iterative searching", "Comprehensive synthesis"],
            "use_case": "Thorough research on complex topics"
        },
        "crewai_research": {
            "endpoint": "/crewai-research",
            "description": "Multi-agent collaborative research with 8 specialized agents",
            "features": [
                "Data-driven question generation", 
                "Collaborative agent workflow",
                "Professional quality control",
                "Comprehensive methodology",
                "Multi-perspective analysis"
            ],
            "use_case": "Complex research requiring multiple expertise areas",
            "available": CREWAI_AVAILABLE,
            "agents": [
                "Supervisor: Generates research questions based on actual data",
                "Analyzer: Creates research methodology", 
                "Student: Searches LlamaIndex database",
                "Developer: Performs computational analysis",
                "Philosopher: Provides strategic guidance",
                "Deep Thinker: Synthesizes findings",
                "Critic: Quality control and evaluation", 
                "Reporter: Creates final report"
            ] if CREWAI_AVAILABLE else []
        }
    }
    
    return {
        "available_methods": methods,
        "recommendation": "Use crewai-research for comprehensive analysis, deep-research for structured investigation, simple-query for quick lookups"
    }
