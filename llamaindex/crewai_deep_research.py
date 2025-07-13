import os
import sys
import logging
from typing import List, Dict, Any, Optional
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
# from crewai_tools import tool  # Not needed for this implementation
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine
from llama_index.core import VectorStoreIndex, load_index_from_storage, StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from dotenv import load_dotenv
import json
import threading

# Configure CrewAI to use OpenAI's best model for complex reasoning
# Note: o1 models have limitations with tools, so we use GPT-4o for tool-heavy agents
CREWAI_MODEL = os.getenv("CREWAI_LLM_MODEL", "gpt-4o")  # Best balance of reasoning + tool support
os.environ["OPENAI_MODEL_NAME"] = CREWAI_MODEL

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# LLM Configuration - Use OpenAI's best reasoning model for different use cases
CREWAI_LLM_MODEL = CREWAI_MODEL  # For CrewAI agents (needs tool support)
LLAMAINDEX_LLM_MODEL = os.getenv("LLAMAINDEX_LLM_MODEL", "gpt-4o")  # For LlamaIndex queries
REASONING_LLM_MODEL = os.getenv("REASONING_LLM_MODEL", "o1-preview")  # For pure reasoning tasks

# Global variables for index access
index = None
index_lock = threading.Lock()

# Database connection setup (from rag_api.py)
def get_engine() -> Engine:
    db_url = os.getenv(
        "DATABASE_URL",
        f"postgresql://{os.getenv('POSTGRES_USER','karakeep')}:{os.getenv('POSTGRES_PASSWORD','karakeep_password')}@{os.getenv('POSTGRES_HOST','postgres')}:{os.getenv('POSTGRES_PORT','5432')}/{os.getenv('POSTGRES_DB','karakeep')}"
    )
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    return create_engine(db_url)

def load_existing_index():
    """Load existing LlamaIndex from storage"""
    global index
    PERSIST_DIR = os.getenv("INDEX_PERSIST_DIR", "./storage")
    embed_model = GoogleGenAIEmbedding(model="text-embedding-004")
    
    try:
        if os.path.exists(PERSIST_DIR) and os.path.exists(os.path.join(PERSIST_DIR, "docstore.json")):
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context, embed_model=embed_model)
            logging.info("LlamaIndex loaded successfully for CrewAI integration")
        else:
            logging.warning("No existing LlamaIndex found for CrewAI integration")
    except Exception as e:
        logging.error(f"Error loading LlamaIndex for CrewAI: {e}")

# Load index on module import
load_existing_index()

# Custom Tools for CrewAI Agents

class LlamaIndexSearchTool(BaseTool):
    name: str = "llamaindex_search"
    description: str = "Search through the indexed bookmark database using vector similarity. Use this to find relevant documents, articles, and content."
    
    def _run(self, query: str) -> str:
        """Execute search using LlamaIndex"""
        global index
        
        with index_lock:
            if index is None:
                return "Error: LlamaIndex not available. Index may not be initialized."
            
            try:
                llm = OpenAI(model=os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini"))
                query_engine = index.as_query_engine(llm=llm)
                response = query_engine.query(query)
                return str(response)
            except Exception as e:
                return f"Search error: {str(e)}"

class DatabaseQueryTool(BaseTool):
    name: str = "database_query"
    description: str = "Query the PostgreSQL database directly to get bookmark metadata, statistics, and raw data. Use for data analysis and exploration."
    
    def _run(self, query_description: str) -> str:
        """Execute database queries based on description"""
        try:
            engine = get_engine()
            with engine.connect() as conn:
                # Map common query descriptions to SQL
                if "count" in query_description.lower() or "total" in query_description.lower():
                    result = conn.execute(sql_text('SELECT COUNT(*) as total FROM "bookmarkLinks"'))
                    count = result.fetchone().total
                    return f"Total bookmarks in database: {count}"
                
                elif "recent" in query_description.lower() or "latest" in query_description.lower():
                    result = conn.execute(sql_text('SELECT id, url, title, LEFT(content, 200) as content_preview FROM "bookmarkLinks" ORDER BY id DESC LIMIT 10'))
                    rows = result.fetchall()
                    output = "Recent bookmarks:\n"
                    for row in rows:
                        output += f"ID: {row.id}, Title: {row.title}, URL: {row.url}\n"
                        if row.content_preview:
                            output += f"Preview: {row.content_preview}...\n"
                        output += "---\n"
                    return output
                
                elif "domains" in query_description.lower() or "sites" in query_description.lower():
                    result = conn.execute(sql_text('''
                        SELECT 
                            SUBSTRING(url FROM 'https?://([^/]+)') as domain,
                            COUNT(*) as count
                        FROM "bookmarkLinks" 
                        WHERE url IS NOT NULL
                        GROUP BY SUBSTRING(url FROM 'https?://([^/]+)')
                        ORDER BY count DESC 
                        LIMIT 15
                    '''))
                    rows = result.fetchall()
                    output = "Top domains in bookmarks:\n"
                    for row in rows:
                        output += f"{row.domain}: {row.count} bookmarks\n"
                    return output
                
                elif "tags" in query_description.lower():
                    result = conn.execute(sql_text('SELECT name, COUNT(*) as usage_count FROM "bookmarkTags" bt JOIN "bookmarks" b ON bt."bookmarkId" = b.id GROUP BY name ORDER BY usage_count DESC LIMIT 20'))
                    rows = result.fetchall()
                    output = "Most used tags:\n"
                    for row in rows:
                        output += f"{row.name}: {row.usage_count} uses\n"
                    return output
                
                else:
                    # General exploration query
                    result = conn.execute(sql_text('SELECT COUNT(*) as total FROM "bookmarkLinks"'))
                    total = result.fetchone().total
                    result = conn.execute(sql_text('SELECT url FROM "bookmarkLinks" WHERE url IS NOT NULL LIMIT 5'))
                    sample_urls = [row.url for row in result.fetchall()]
                    
                    return f"Database overview: {total} total bookmarks. Sample URLs: {', '.join(sample_urls[:3])}..."
        
        except Exception as e:
            return f"Database query error: {str(e)}"

class CalculationTool(BaseTool):
    name: str = "calculation"
    description: str = "Perform mathematical calculations, data analysis, and computational tasks. Provide the calculation or analysis you need."
    
    def _run(self, calculation: str) -> str:
        """Perform calculations safely"""
        try:
            # Simple safe evaluation for basic math
            if any(dangerous in calculation.lower() for dangerous in ['import', 'exec', 'eval', '__', 'open', 'file']):
                return "Error: Unsafe calculation request"
            
            # Handle basic mathematical expressions
            import re
            import math
            
            # Replace common math functions
            safe_calculation = calculation.replace('^', '**')
            
            # Allow basic math operations
            allowed_chars = set('0123456789+-*/.() \t\n')
            if all(c in allowed_chars for c in safe_calculation):
                try:
                    result = eval(safe_calculation)
                    return f"Calculation result: {result}"
                except:
                    pass
            
            # For more complex analysis, provide guidance
            return f"For calculation '{calculation}': Please provide specific numerical data or mathematical expressions. Available functions: basic arithmetic, percentages, averages."
            
        except Exception as e:
            return f"Calculation error: {str(e)}"

class DataExplorationTool(BaseTool):
    name: str = "data_exploration"
    description: str = "Explore patterns, trends, and insights in the bookmark data. Use for discovering interesting data characteristics."
    
    def _run(self, exploration_type: str) -> str:
        """Explore data patterns"""
        try:
            engine = get_engine()
            with engine.connect() as conn:
                if "pattern" in exploration_type.lower() or "trend" in exploration_type.lower():
                    # Analyze bookmark creation patterns
                    result = conn.execute(sql_text('''
                        SELECT 
                            DATE_TRUNC('month', "crawledAt") as month,
                            COUNT(*) as bookmark_count
                        FROM "bookmarkLinks"
                        WHERE "crawledAt" IS NOT NULL
                        GROUP BY DATE_TRUNC('month', "crawledAt")
                        ORDER BY month DESC
                        LIMIT 12
                    '''))
                    rows = result.fetchall()
                    output = "Bookmark creation trends (last 12 months):\n"
                    for row in rows:
                        output += f"{row.month.strftime('%Y-%m')}: {row.bookmark_count} bookmarks\n"
                    return output
                
                elif "content" in exploration_type.lower():
                    # Analyze content characteristics
                    result = conn.execute(sql_text('''
                        SELECT 
                            CASE 
                                WHEN LENGTH(content) = 0 OR content IS NULL THEN 'No content'
                                WHEN LENGTH(content) < 500 THEN 'Short content'
                                WHEN LENGTH(content) < 2000 THEN 'Medium content'
                                ELSE 'Long content'
                            END as content_type,
                            COUNT(*) as count
                        FROM "bookmarkLinks"
                        GROUP BY 
                            CASE 
                                WHEN LENGTH(content) = 0 OR content IS NULL THEN 'No content'
                                WHEN LENGTH(content) < 500 THEN 'Short content'
                                WHEN LENGTH(content) < 2000 THEN 'Medium content'
                                ELSE 'Long content'
                            END
                        ORDER BY count DESC
                    '''))
                    rows = result.fetchall()
                    output = "Content length distribution:\n"
                    for row in rows:
                        output += f"{row.content_type}: {row.count} bookmarks\n"
                    return output
                
                else:
                    return "Available exploration types: 'pattern/trend' for temporal analysis, 'content' for content characteristics"
                    
        except Exception as e:
            return f"Data exploration error: {str(e)}"

class SQLiteDocumentAnalysisTool(BaseTool):
    name: str = "sqlite_document_analysis"
    description: str = "Perform DEEP multi-layered analysis of SQLite documents (English/Chinese/Japanese) to identify highly specific research gaps, micro-trends, cross-document patterns, and truly novel opportunities requiring original investigation."
    
    def _detect_language_patterns(self, text: str) -> dict:
        """Detect language patterns in text and return language-specific keywords"""
        if not text:
            return {"language": "unknown", "keywords": []}
        
        # Simple language detection based on character patterns
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        japanese_chars = len([c for c in text if '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff'])
        total_chars = len(text)
        
        if chinese_chars / total_chars > 0.1:
            language = "chinese"
            keywords = {
                "tech": ["算法", "模型", "系统", "方法", "框架", "分析", "研究", "学习", "智能", "技术", "创新", "突破"],
                "novelty": ["突破", "创新", "首次", "新颖", "革命性", "前沿", "颠覆性", "变革性", "开创性"],
                "research": ["研究", "调查", "实验", "分析", "探索", "发现", "方法论", "理论", "实证"]
            }
        elif japanese_chars / total_chars > 0.1:
            language = "japanese"
            keywords = {
                "tech": ["アルゴリズム", "モデル", "システム", "手法", "フレームワーク", "分析", "研究", "学習", "知能", "技術", "革新"],
                "novelty": ["画期的", "革新的", "初回", "新規", "革命的", "最先端", "破壊的", "変革的"],
                "research": ["研究", "調査", "実験", "分析", "探索", "発見", "方法論", "理論", "実証"]
            }
        else:
            language = "english"
            keywords = {
                "tech": ["algorithm", "model", "system", "method", "approach", "framework", "analysis", "research", "study", "investigation", "experiment", "artificial", "machine", "neural", "deep", "learning", "intelligence", "technology", "innovation"],
                "novelty": ["breakthrough", "novel", "first", "unprecedented", "revolutionary", "cutting edge", "groundbreaking", "disruptive", "game changing", "transformative", "paradigm", "emerging"],
                "research": ["research", "study", "investigation", "analysis", "exploration", "discovery", "methodology", "theory", "empirical", "evidence"]
            }
        
        return {"language": language, "keywords": keywords}
    
    def _run(self, analysis_focus: str) -> str:
        """Analyze documents for research opportunities with multi-language support"""
        try:
            engine = get_engine()
            with engine.connect() as conn:
                # Comprehensive document analysis for research question generation
                
                if "themes" in analysis_focus.lower() or "topics" in analysis_focus.lower():
                    # Multi-language theme analysis
                    result = conn.execute(sql_text('''
                        SELECT 
                            title,
                            content,
                            url,
                            "crawledAt"
                        FROM "bookmarkLinks"
                        WHERE content IS NOT NULL 
                        AND LENGTH(content) > 200
                        ORDER BY "crawledAt" DESC
                        LIMIT 80
                    '''))
                    rows = result.fetchall()
                    
                    output = "MULTI-LANGUAGE DOCUMENT THEME ANALYSIS:\n"
                    output += f"Analyzed {len(rows)} recent substantial documents\n\n"
                    
                    # Language-aware theme categorization
                    language_stats = {"english": 0, "chinese": 0, "japanese": 0, "unknown": 0}
                    tech_themes = {"english": [], "chinese": [], "japanese": []}
                    research_themes = {"english": [], "chinese": [], "japanese": []}
                    novel_themes = {"english": [], "chinese": [], "japanese": []}
                    
                    for row in rows:
                        # Detect language for each document
                        combined_text = f"{row.title or ''} {row.content[:500] if row.content else ''}"
                        lang_info = self._detect_language_patterns(combined_text)
                        language = lang_info["language"]
                        keywords = lang_info["keywords"]
                        
                        language_stats[language] += 1
                        
                        if language in ["english", "chinese", "japanese"]:
                            title_text = (row.title or "").lower()
                            content_text = (row.content or "").lower()
                            
                            # Check for tech themes using language-specific keywords
                            if any(keyword in title_text or keyword in content_text for keyword in keywords.get("tech", [])):
                                tech_themes[language].append(row.title or "Untitled")
                            
                            # Check for research themes
                            if any(keyword in title_text or keyword in content_text for keyword in keywords.get("research", [])):
                                research_themes[language].append(row.title or "Untitled")
                            
                            # Check for novelty themes
                            if any(keyword in title_text or keyword in content_text for keyword in keywords.get("novelty", [])):
                                novel_themes[language].append(row.title or "Untitled")
                    
                    # Report findings by language
                    output += f"LANGUAGE DISTRIBUTION:\n"
                    for lang, count in language_stats.items():
                        if count > 0:
                            output += f"• {lang.title()}: {count} documents\n"
                    
                    output += f"\nTECHNOLOGY THEMES BY LANGUAGE:\n"
                    for lang in ["english", "chinese", "japanese"]:
                        if tech_themes[lang]:
                            output += f"• {lang.title()} ({len(tech_themes[lang])}): {', '.join(tech_themes[lang][:3])}...\n"
                    
                    output += f"\nRESEARCH THEMES BY LANGUAGE:\n"
                    for lang in ["english", "chinese", "japanese"]:
                        if research_themes[lang]:
                            output += f"• {lang.title()} ({len(research_themes[lang])}): {', '.join(research_themes[lang][:3])}...\n"
                    
                    output += f"\nNOVELTY THEMES BY LANGUAGE:\n"
                    for lang in ["english", "chinese", "japanese"]:
                        if novel_themes[lang]:
                            output += f"• {lang.title()} ({len(novel_themes[lang])}): {', '.join(novel_themes[lang][:3])}...\n"
                    
                    return output
                
                elif "gaps" in analysis_focus.lower() or "opportunities" in analysis_focus.lower():
                    # Identify research gaps and opportunities
                    result = conn.execute(sql_text('''
                        SELECT 
                            SUBSTRING(url FROM 'https?://([^/]+)') as domain,
                            COUNT(*) as count,
                            STRING_AGG(DISTINCT title, ' | ') as sample_titles
                        FROM "bookmarkLinks" 
                        WHERE url IS NOT NULL
                        GROUP BY SUBSTRING(url FROM 'https?://([^/]+)')
                        HAVING COUNT(*) >= 3
                        ORDER BY count DESC 
                        LIMIT 15
                    '''))
                    rows = result.fetchall()
                    
                    output = "RESEARCH GAP ANALYSIS:\n"
                    output += "Domains with multiple bookmarks (potential research focus areas):\n\n"
                    
                    for row in rows:
                        output += f"• {row.domain} ({row.count} bookmarks)\n"
                        sample_titles = row.sample_titles.split(' | ')[:3]
                        output += f"  Sample topics: {', '.join(sample_titles)}\n\n"
                    
                    # Identify underrepresented areas
                    result = conn.execute(sql_text('''
                        SELECT 
                            DATE_TRUNC('week', "crawledAt") as week,
                            COUNT(*) as weekly_count
                        FROM "bookmarkLinks"
                        WHERE "crawledAt" >= NOW() - INTERVAL '12 weeks'
                        GROUP BY DATE_TRUNC('week', "crawledAt")
                        ORDER BY week DESC
                    '''))
                    weekly_data = result.fetchall()
                    
                    if weekly_data:
                        avg_weekly = sum(row.weekly_count for row in weekly_data) / len(weekly_data)
                        output += f"\nRecent activity: Average {avg_weekly:.1f} bookmarks/week\n"
                        output += "This suggests active curation in these areas.\n"
                    
                    return output
                
                elif "novelty" in analysis_focus.lower() or "unique" in analysis_focus.lower():
                    # Deep novelty analysis with cross-document pattern detection
                    result = conn.execute(sql_text('''
                        SELECT 
                            title,
                            content,
                            url,
                            "crawledAt"
                        FROM "bookmarkLinks"
                        WHERE content IS NOT NULL
                        AND (
                            title ILIKE '%breakthrough%' OR
                            title ILIKE '%novel%' OR 
                            title ILIKE '%first%' OR
                            title ILIKE '%emerging%' OR
                            title ILIKE '%revolutionary%' OR
                            title ILIKE '%paradigm%' OR
                            title ILIKE '%unprecedented%' OR
                            content ILIKE '%never before%' OR
                            content ILIKE '%first time%' OR
                            content ILIKE '%cutting edge%' OR
                            content ILIKE '%groundbreaking%' OR
                            content ILIKE '%disruptive%' OR
                            content ILIKE '%game changing%' OR
                            content ILIKE '%transformative%'
                        )
                        ORDER BY "crawledAt" DESC
                        LIMIT 30
                    '''))
                    rows = result.fetchall()
                    
                    # Advanced novelty analysis
                    output = "DEEP NOVELTY ANALYSIS:\n"
                    output += f"Analyzed {len(rows)} documents with high novelty indicators\n\n"
                    
                    # Categorize by novelty type
                    tech_breakthroughs = []
                    method_innovations = []
                    paradigm_shifts = []
                    emerging_fields = []
                    
                    for row in rows:
                        title_lower = row.title.lower() if row.title else ""
                        content_lower = row.content.lower() if row.content else ""
                        
                        if any(word in title_lower or word in content_lower for word in ['breakthrough', 'revolutionary', 'groundbreaking']):
                            tech_breakthroughs.append(row)
                        elif any(word in title_lower or word in content_lower for word in ['method', 'approach', 'technique', 'algorithm']):
                            method_innovations.append(row)
                        elif any(word in title_lower or word in content_lower for word in ['paradigm', 'shift', 'transformation', 'disruption']):
                            paradigm_shifts.append(row)
                        else:
                            emerging_fields.append(row)
                    
                    output += f"NOVELTY CATEGORIES:\n"
                    output += f"• Technical Breakthroughs: {len(tech_breakthroughs)} documents\n"
                    output += f"• Methodological Innovations: {len(method_innovations)} documents\n"
                    output += f"• Paradigm Shifts: {len(paradigm_shifts)} documents\n"
                    output += f"• Emerging Fields: {len(emerging_fields)} documents\n\n"
                    
                    # Show top novelty examples
                    output += "HIGH-NOVELTY RESEARCH OPPORTUNITIES:\n"
                    all_novel = tech_breakthroughs + method_innovations + paradigm_shifts + emerging_fields
                    for i, row in enumerate(all_novel[:8], 1):
                        output += f"{i}. {row.title}\n"
                        if row.content:
                            # Extract key novelty phrases
                            content_words = row.content.lower().split()
                            novelty_phrases = []
                            for j, word in enumerate(content_words):
                                if word in ['breakthrough', 'novel', 'first', 'unprecedented', 'revolutionary']:
                                    phrase = ' '.join(content_words[max(0,j-3):j+4])
                                    novelty_phrases.append(phrase)
                            
                            if novelty_phrases:
                                output += f"   Key novelty: {novelty_phrases[0][:100]}...\n"
                            else:
                                novelty_snippet = row.content[:120] + "..." if len(row.content) > 120 else row.content
                                output += f"   Content: {novelty_snippet}\n"
                        output += f"   Date: {row.crawledAt.strftime('%Y-%m-%d') if row.crawledAt else 'Unknown'}\n\n"
                    
                    return output
                
                elif "deep" in analysis_focus.lower() or "cross" in analysis_focus.lower():
                    # Cross-document pattern analysis for deep insights
                    result = conn.execute(sql_text('''
                        SELECT 
                            title,
                            content,
                            url,
                            "crawledAt",
                            LENGTH(content) as content_length
                        FROM "bookmarkLinks"
                        WHERE content IS NOT NULL 
                        AND LENGTH(content) > 500
                        ORDER BY "crawledAt" DESC
                        LIMIT 100
                    '''))
                    rows = result.fetchall()
                    
                    output = "CROSS-DOCUMENT PATTERN ANALYSIS:\n"
                    output += f"Deep analysis of {len(rows)} substantial documents\n\n"
                    
                    # Extract recurring themes and contradictions
                    keyword_frequency = {}
                    concept_clusters = {}
                    contradictions = []
                    
                    for row in rows:
                        if row.content:
                            words = row.content.lower().split()
                            # Count technical terms
                            tech_terms = ['algorithm', 'model', 'system', 'method', 'approach', 'framework', 
                                        'analysis', 'research', 'study', 'investigation', 'experiment',
                                        'artificial', 'machine', 'neural', 'deep', 'learning', 'intelligence']
                            
                            for term in tech_terms:
                                if term in ' '.join(words):
                                    keyword_frequency[term] = keyword_frequency.get(term, 0) + 1
                    
                    # Identify research gaps from patterns
                    output += "DEEP PATTERN INSIGHTS:\n"
                    top_concepts = sorted(keyword_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
                    output += f"Most frequent concepts: {', '.join([f'{k}({v})' for k,v in top_concepts])}\n\n"
                    
                    # Find underexplored intersections
                    output += "UNDEREXPLORED INTERSECTIONS:\n"
                    intersections = []
                    concept_list = [k for k,v in top_concepts[:5]]
                    for i, concept1 in enumerate(concept_list):
                        for concept2 in concept_list[i+1:]:
                            # Check how often these concepts appear together
                            together_count = 0
                            for row in rows:
                                if row.content and concept1 in row.content.lower() and concept2 in row.content.lower():
                                    together_count += 1
                            if together_count < 3:  # Underexplored combination
                                intersections.append(f"{concept1} + {concept2} (only {together_count} documents)")
                    
                    for intersection in intersections[:5]:
                        output += f"• {intersection}\n"
                    
                    output += f"\nDEEP RESEARCH GAPS IDENTIFIED:\n"
                    output += f"• Temporal analysis: Recent vs historical document patterns\n"
                    output += f"• Methodological gaps: Missing research approaches\n"
                    output += f"• Cross-domain opportunities: Unexplored field combinations\n"
                    output += f"• Empirical gaps: Theory-heavy vs evidence-light areas\n"
                    
                    return output
                
                else:
                    # Default comprehensive analysis
                    result = conn.execute(sql_text('''
                        SELECT COUNT(*) as total FROM "bookmarkLinks"
                    '''))
                    total = result.fetchone().total
                    
                    result = conn.execute(sql_text('''
                        SELECT COUNT(*) as with_content 
                        FROM "bookmarkLinks" 
                        WHERE content IS NOT NULL AND LENGTH(content) > 100
                    '''))
                    with_content = result.fetchone().with_content
                    
                    output = f"DOCUMENT COLLECTION OVERVIEW:\n"
                    output += f"Total documents: {total}\n"
                    output += f"Documents with substantial content: {with_content}\n"
                    output += f"Content coverage: {(with_content/total*100):.1f}%\n\n"
                    output += "Available analysis types:\n"
                    output += "- 'themes' - Extract content themes and topics\n"
                    output += "- 'gaps' - Identify research gaps and opportunities\n"
                    output += "- 'novelty' - Find unique and novel content patterns\n"
                    
                    return output
                    
        except Exception as e:
            return f"SQLite document analysis error: {str(e)}"

class HybridAnalysisTool(BaseTool):
    name: str = "hybrid_analysis"
    description: str = "Combine SQLite document analysis with LlamaIndex RAG retrieval for comprehensive multi-language research insights. Use this for deep cross-referencing between direct database patterns and semantic similarity searches."
    
    def _run(self, research_focus: str) -> str:
        """Perform hybrid analysis combining SQLite patterns with RAG retrieval"""
        try:
            output = "HYBRID ANALYSIS: SQLite + LlamaIndex RAG\n"
            output += f"Research Focus: {research_focus}\n"
            output += "=" * 60 + "\n\n"
            
            # Phase 1: SQLite Pattern Analysis
            sqlite_tool = SQLiteDocumentAnalysisTool()
            
            output += "PHASE 1: SQLite Document Pattern Analysis\n"
            output += "-" * 40 + "\n"
            
            # Get themes and novelty patterns from SQLite
            themes_analysis = sqlite_tool._run("themes")
            novelty_analysis = sqlite_tool._run("novelty")
            
            output += f"Themes Analysis:\n{themes_analysis[:300]}...\n\n"
            output += f"Novelty Analysis:\n{novelty_analysis[:300]}...\n\n"
            
            # Phase 2: LlamaIndex RAG Retrieval
            output += "PHASE 2: LlamaIndex RAG Semantic Retrieval\n"
            output += "-" * 40 + "\n"
            
            global index
            with index_lock:
                if index is not None:
                    try:
                        from llama_index.llms.openai import OpenAI
                        llm = OpenAI(model=LLAMAINDEX_LLM_MODEL, temperature=0.3)
                        query_engine = index.as_query_engine(llm=llm)
                        
                        # Perform multiple RAG queries for different aspects
                        rag_queries = [
                            f"{research_focus} emerging trends",
                            f"{research_focus} novel approaches", 
                            f"{research_focus} research gaps",
                            f"{research_focus} methodological innovations"
                        ]
                        
                        rag_results = []
                        for i, query in enumerate(rag_queries, 1):
                            try:
                                response = query_engine.query(query)
                                rag_results.append({
                                    "query": query,
                                    "response": str(response)
                                })
                                output += f"RAG Query {i}: {query}\n"
                                output += f"Result: {str(response)[:200]}...\n\n"
                            except Exception as e:
                                output += f"RAG Query {i} failed: {str(e)}\n"
                        
                        # Phase 3: Cross-Analysis and Synthesis
                        output += "PHASE 3: Cross-Analysis & Synthesis\n"
                        output += "-" * 40 + "\n"
                        
                        # Identify convergent themes between SQLite and RAG
                        output += "CONVERGENT INSIGHTS:\n"
                        output += "• SQLite reveals document patterns and multi-language distribution\n"
                        output += "• RAG provides semantic similarity and contextual relationships\n"
                        output += "• Hybrid approach identifies both explicit and implicit research opportunities\n\n"
                        
                        # Language-specific RAG insights
                        if "chinese" in themes_analysis.lower() or "japanese" in themes_analysis.lower():
                            output += "MULTI-LANGUAGE CONSIDERATIONS:\n"
                            output += "• Cross-language research opportunities detected\n"
                            output += "• Consider translation and cultural context factors\n"
                            output += "• Potential for comparative analysis across language domains\n\n"
                        
                        # Research opportunity synthesis
                        output += "SYNTHESIZED RESEARCH OPPORTUNITIES:\n"
                        output += "• Combine SQLite pattern gaps with RAG semantic relationships\n"
                        output += "• Cross-validate novelty claims through multiple analysis methods\n"
                        output += "• Identify research questions that bridge database patterns and semantic insights\n"
                        
                    except Exception as e:
                        output += f"LlamaIndex RAG error: {str(e)}\n"
                        output += "Falling back to SQLite-only analysis.\n"
                else:
                    output += "LlamaIndex not available, using SQLite analysis only.\n"
                    
                    # Enhanced SQLite analysis when RAG unavailable
                    output += "\nENHANCED SQLite ANALYSIS:\n"
                    cross_analysis = sqlite_tool._run("deep")
                    output += cross_analysis[:400] + "...\n"
            
            return output
            
        except Exception as e:
            return f"Hybrid analysis error: {str(e)}"

class AdvancedReasoningTool(BaseTool):
    name: str = "advanced_reasoning"
    description: str = "Use OpenAI's most advanced reasoning model (o1-preview) for complex analytical tasks requiring deep logical reasoning, pattern synthesis, and novel insight generation."
    
    def _run(self, reasoning_task: str) -> str:
        """Perform advanced reasoning using OpenAI's o1-preview model"""
        try:
            from llama_index.llms.openai import OpenAI
            
            # Use o1-preview for maximum reasoning capability
            reasoning_llm = OpenAI(
                model=REASONING_LLM_MODEL,
                temperature=1.0,  # o1 models use higher temperature for reasoning
                max_tokens=4000   # Allow for detailed reasoning
            )
            
            # Enhanced prompt for complex reasoning
            reasoning_prompt = f"""
            ADVANCED REASONING TASK:
            {reasoning_task}
            
            Please apply deep logical reasoning, pattern analysis, and synthesis to provide novel insights.
            Consider multiple perspectives, potential contradictions, and innovative connections.
            Provide detailed reasoning steps and justify your conclusions.
            """
            
            response = reasoning_llm.complete(reasoning_prompt)
            
            output = f"ADVANCED REASONING OUTPUT (Model: {REASONING_LLM_MODEL}):\n"
            output += "=" * 60 + "\n"
            output += f"Task: {reasoning_task}\n\n"
            output += f"Reasoning:\n{str(response)}\n"
            
            return output
            
        except Exception as e:
            # Fallback to regular model if o1-preview fails
            try:
                from llama_index.llms.openai import OpenAI
                fallback_llm = OpenAI(model=LLAMAINDEX_LLM_MODEL, temperature=0.8)
                response = fallback_llm.complete(f"Advanced reasoning task: {reasoning_task}")
                
                output = f"REASONING OUTPUT (Fallback Model: {LLAMAINDEX_LLM_MODEL}):\n"
                output += "=" * 60 + "\n"
                output += f"Note: o1-preview unavailable, using fallback\n"
                output += f"Task: {reasoning_task}\n\n"
                output += f"Analysis:\n{str(response)}\n"
                
                return output
            except Exception as e2:
                return f"Advanced reasoning error: {str(e)} | Fallback error: {str(e2)}"

# Agent Definitions

def create_supervisor_agent() -> Agent:
    """Research Supervisor: Orchestrates research and delegates document analysis to Developer"""
    return Agent(
        role='Research Supervisor',
        goal='Orchestrate comprehensive research by delegating SQLite document analysis to the Developer, then generate targeted research questions based on the analysis',
        backstory="""You are a senior research supervisor who excels at coordinating research teams. You understand that thorough document analysis is crucial for generating meaningful research questions. 
        You delegate the technical analysis of SQLite documents to the Developer agent, then use their findings to craft research questions that are both data-driven and strategically important.
        You ensure all research questions have strong empirical foundations based on actual document analysis.""",
        tools=[DatabaseQueryTool()],  # Basic tools only - delegates complex analysis
        verbose=True,
        allow_delegation=True
    )

def create_analyzer_agent() -> Agent:
    """Strategic Analyzer: Creates comprehensive research plans"""
    return Agent(
        role='Strategic Analyzer',
        goal='Design comprehensive, iterative, and flexible research methodologies and plans',
        backstory="""You are a strategic research analyst who excels at creating structured approaches to complex research problems.
        You design methodologies that are both comprehensive and adaptable, ensuring research efforts are well-organized and productive.
        You think systematically about how different research activities should be sequenced and coordinated.""",
        tools=[DatabaseQueryTool()],
        verbose=True,
        allow_delegation=True
    )

def create_student_agent() -> Agent:
    """Research Student: Searches through LlamaIndex"""
    return Agent(
        role='Research Student',
        goal='Search through the indexed bookmark database to find relevant information and gather research materials',
        backstory="""You are a diligent research student who excels at finding relevant information in large databases.
        You know how to formulate effective search queries and can quickly identify the most relevant sources for any research question.
        You are thorough in your search efforts and always provide comprehensive results.""",
        tools=[LlamaIndexSearchTool(), DatabaseQueryTool()],
        verbose=True,
        allow_delegation=False
    )

def create_developer_agent() -> Agent:
    """Developer: Hybrid deep analysis using SQLite + LlamaIndex RAG"""
    return Agent(
        role='Hybrid Research Developer',
        goal='Perform DUAL-MODE analysis combining SQLite document pattern analysis with LlamaIndex RAG retrieval for multi-language (English/Chinese/Japanese) research insights and cross-validation of findings',
        backstory="""You are an elite hybrid research developer who masters both direct database analysis and semantic retrieval systems.
        
        Your DUAL-MODE approach:
        1. SQLite ANALYSIS: Direct document pattern mining, multi-language keyword detection, temporal trends, cross-document relationships
        2. RAG RETRIEVAL: Semantic similarity search, contextual understanding, implicit relationship discovery
        
        You excel at:
        - Multi-language research (English/Chinese/Japanese) with proper encoding handling
        - Cross-validating findings between database patterns and semantic retrieval
        - Identifying research opportunities that require both explicit patterns and implicit relationships
        - Bridging language barriers to find cross-cultural research opportunities
        - Combining structured database insights with unstructured semantic understanding
        
        Your hybrid analysis reveals research territories invisible to single-mode approaches.""",
        tools=[SQLiteDocumentAnalysisTool(), HybridAnalysisTool(), LlamaIndexSearchTool(), CalculationTool(), DataExplorationTool(), DatabaseQueryTool()],
        verbose=True,
        allow_delegation=False
    )

def create_philosopher_agent() -> Agent:
    """Philosopher: Provides broader scope and strategic direction"""
    return Agent(
        role='Research Philosopher',
        goal='Guide the broader scope and strategic direction of research, providing philosophical insights and context',
        backstory="""You are a research philosopher who thinks deeply about the broader implications and contexts of research topics.
        You excel at stepping back from details to see the bigger picture and can provide valuable perspective on research direction.
        You ask profound questions and help ensure research efforts are meaningful and well-contextualized.""",
        tools=[],
        verbose=True,
        allow_delegation=True
    )

def create_deep_thinker_agent() -> Agent:
    """Deep Thinker: Advanced reasoning and synthesis using o1-preview"""
    return Agent(
        role='Deep Thinker',
        goal='Apply advanced reasoning to synthesize findings, identify complex patterns, draw novel connections, and generate breakthrough insights using the most sophisticated reasoning models',
        backstory="""You are an elite analytical thinker with access to the most advanced reasoning capabilities.
        You excel at synthesis, pattern recognition, and logical reasoning at the highest level.
        You can take disparate pieces of information and weave them together into revolutionary insights.
        You use advanced reasoning models to think deeply about evidence, identify hidden patterns, and draw novel conclusions.
        Your insights often reveal connections and implications that others miss entirely.""",
        tools=[AdvancedReasoningTool(), DataExplorationTool()],
        verbose=True,
        allow_delegation=False
    )

def create_reporter_agent() -> Agent:
    """Reporter: Creates concise final reports"""
    return Agent(
        role='Research Reporter',
        goal='Create concise, clear, and actionable research reports that communicate findings effectively',
        backstory="""You are a skilled research reporter who excels at distilling complex research into clear, actionable insights.
        You know how to communicate findings in a way that is both comprehensive and accessible.
        You focus on practical implications and ensure that research results are presented in a useful format.""",
        tools=[],
        verbose=True,
        allow_delegation=False
    )

def create_critic_agent() -> Agent:
    """Elite Research Critic: Enforces highest novelty and depth standards"""
    return Agent(
        role='Elite Research Critic',
        goal='Enforce MAXIMUM novelty standards (≥8.5/10), research depth requirements, and specificity criteria. Reject anything generic, shallow, or insufficiently novel',
        backstory="""You are an elite research critic from top-tier institutions who maintains the highest standards for groundbreaking research.
        You REJECT research questions that are generic, shallow, or lack true novelty. Your standards are extremely high - only truly innovative, deep, and specific research passes your review.
        
        NOVELTY REQUIREMENTS (≥8.5/10):
        - Must explore genuinely unexplored territory
        - Must combine concepts in unprecedented ways  
        - Must challenge existing paradigms or assumptions
        - Must have potential for paradigm-shifting insights
        
        DEPTH REQUIREMENTS:
        - Must require multi-layered investigation
        - Must demand sophisticated analytical approaches
        - Must go beyond surface-level observations
        
        SPECIFICITY REQUIREMENTS:
        - Must be precisely focused, not broad generalizations
        - Must target specific mechanisms, patterns, or phenomena
        - Must be actionable and measurable
        
        You provide harsh but constructive criticism to elevate research to world-class standards.""",
        tools=[DatabaseQueryTool(), DataExplorationTool(), SQLiteDocumentAnalysisTool(), AdvancedReasoningTool()],
        verbose=True,
        allow_delegation=True
    )

class CrewAIDeepResearch:
    """CrewAI-based Deep Research Multi-Agent System with Advanced Reasoning LLM"""
    
    def __init__(self):
        # CrewAI agents will use the model configured in OPENAI_MODEL_NAME environment variable
        # which is set to gpt-4o for best balance of reasoning and tool support
        self.crewai_model = CREWAI_LLM_MODEL
        self.llamaindex_model = LLAMAINDEX_LLM_MODEL
        self.reasoning_model = REASONING_LLM_MODEL
        
        logging.info(f"CrewAI System initialized with:")
        logging.info(f"  • CrewAI Agents: {self.crewai_model}")
        logging.info(f"  • LlamaIndex Queries: {self.llamaindex_model}")
        logging.info(f"  • Pure Reasoning: {self.reasoning_model}")
        
        self.agents = self._create_agents()
        
    def _create_agents(self) -> Dict[str, Agent]:
        """Create all research agents"""
        return {
            'supervisor': create_supervisor_agent(),
            'analyzer': create_analyzer_agent(),
            'student': create_student_agent(),
            'developer': create_developer_agent(),
            'philosopher': create_philosopher_agent(),
            'deep_thinker': create_deep_thinker_agent(),
            'reporter': create_reporter_agent(),
            'critic': create_critic_agent()
        }
    
    def _create_tasks(self, research_query: str) -> List[Task]:
        """Create research tasks with flexible workflow implementing new constraints"""
        
        # Task 1: Developer performs HYBRID multi-language deep analysis
        developer_analysis_task = Task(
            description=f"""
            Perform HYBRID DUAL-MODE analysis for: '{research_query}'
            
            REQUIRED HYBRID APPROACH:
            1. SQLite DOCUMENT ANALYSIS (Multi-language: English/Chinese/Japanese)
               - Use sqlite_document_analysis tool with modes: 'themes', 'novelty', 'deep', 'gaps'
               - Detect language patterns and cross-language research opportunities
               - Identify document patterns, temporal trends, and explicit relationships
            
            2. LlamaIndex RAG RETRIEVAL
               - Use hybrid_analysis tool to combine SQLite + RAG insights
               - Perform semantic similarity searches with llamaindex_search tool
               - Cross-validate findings between database patterns and semantic understanding
            
            3. CROSS-VALIDATION & SYNTHESIS
               - Compare SQLite explicit patterns with RAG implicit relationships
               - Identify convergent insights and contradictory findings
               - Find research opportunities that bridge database patterns and semantic insights
            
            MULTI-LANGUAGE REQUIREMENTS:
            - Analyze content in English, Chinese, and Japanese
            - Identify cross-language research opportunities
            - Consider cultural context and translation factors
            - Find comparative analysis possibilities across language domains
            
            DEPTH REQUIREMENTS:
            - Use BOTH direct database mining AND semantic retrieval
            - Cross-validate novelty claims through multiple analysis methods
            - Identify specific research gaps invisible to single-mode approaches
            - Find micro-trends and hidden connections across analysis modes
            
            Your hybrid analysis must reveal multi-dimensional research opportunities requiring groundbreaking cross-modal investigation.
            """,
            agent=self.agents['developer'],
            expected_output="Comprehensive hybrid analysis (SQLite + RAG) revealing multi-language, cross-validated research opportunities with explicit and implicit insights"
        )
        
        # Task 2: Supervisor generates research questions based on Developer's analysis
        supervisor_task = Task(
            description=f"""
            Based on the Developer's comprehensive SQLite document analysis, generate 3-5 research questions for: '{research_query}'
            
            Use the Developer's findings to create questions that:
            1. Target identified research gaps and opportunities
            2. Leverage areas with sufficient document coverage
            3. Focus on novel or emerging themes found in the analysis
            4. Are specific, focused, and data-driven
            5. Build upon each other logically
            
            For each question, explain:
            - How it addresses a specific gap or opportunity identified
            - Why it's important based on the document analysis
            - What makes it feasible given available data
            
            Delegate to the Developer if you need additional technical analysis.
            """,
            agent=self.agents['supervisor'],
            expected_output="3-5 data-driven research questions with detailed justifications",
            context=[developer_analysis_task]
        )
        
        # Task 2: Analyzer creates research methodology
        analyzer_task = Task(
            description=f"""
            Based on the research questions generated by the Supervisor, create a comprehensive research methodology.
            
            Design a flexible research plan that outlines:
            1. How each research question should be approached
            2. What types of searches and analysis are needed
            3. How findings should be synthesized
            4. Quality control measures
            
            Make the plan adaptive and iterative, allowing for discoveries to shape the research direction.
            """,
            agent=self.agents['analyzer'],
            expected_output="A comprehensive research methodology and plan",
            context=[supervisor_task]
        )
        
        # Task 3: Student searches for information
        student_task = Task(
            description=f"""
            Using the research questions and methodology provided, conduct thorough searches through the bookmark database.
            
            For each research question:
            1. Formulate effective search queries
            2. Search through the indexed content
            3. Gather relevant information and sources
            4. Organize findings by research question
            
            Be thorough and systematic in your search approach.
            """,
            agent=self.agents['student'],
            expected_output="Comprehensive search results organized by research question",
            context=[supervisor_task, analyzer_task]
        )
        
        # Task 4: Developer performs analysis
        developer_task = Task(
            description=f"""
            Analyze the data and perform any computational tasks needed to support the research.
            
            Tasks may include:
            1. Analyzing patterns in the bookmark data
            2. Performing calculations related to findings
            3. Exploring data characteristics
            4. Computing statistics or metrics
            
            Focus on quantitative insights that complement the qualitative research.
            """,
            agent=self.agents['developer'],
            expected_output="Computational analysis and data insights",
            context=[supervisor_task, student_task]
        )
        
        # Task 5: Philosopher provides broader context
        philosopher_task = Task(
            description=f"""
            Review the research questions, methodology, and initial findings to provide broader philosophical context.
            
            Consider:
            1. The deeper implications of the research topic
            2. Whether the research direction is meaningful
            3. What broader questions or contexts should be considered
            4. How this research fits into larger frameworks of knowledge
            
            Provide strategic guidance for the research direction.
            """,
            agent=self.agents['philosopher'],
            expected_output="Philosophical context and strategic guidance",
            context=[supervisor_task, analyzer_task, student_task]
        )
        
        # Task 6: Deep Thinker synthesizes findings
        synthesis_task = Task(
            description=f"""
            Synthesize all the research findings, analysis, and insights gathered so far.
            
            Create a comprehensive synthesis that:
            1. Identifies key patterns and themes
            2. Draws connections between different findings
            3. Develops insights and conclusions
            4. Identifies gaps or areas needing further investigation
            
            Think deeply about what the evidence tells us and what it means.
            """,
            agent=self.agents['deep_thinker'],
            expected_output="Comprehensive synthesis of all findings and insights",
            context=[student_task, developer_task, philosopher_task]
        )
        
        # Task 3: Elite Critic enforces MAXIMUM novelty and depth standards
        novelty_critique_task = Task(
            description=f"""
            ENFORCE MAXIMUM RESEARCH STANDARDS. Evaluate research questions with EXTREME RIGOR.
            
            MINIMUM REQUIREMENTS (REJECT if not met):
            1. NOVELTY SCORE ≥8.5/10 - Must be truly groundbreaking, not incremental
            2. DEPTH REQUIREMENT - Must require multi-layered, sophisticated investigation
            3. SPECIFICITY REQUIREMENT - Must be precisely focused, not generic
            4. PARADIGM POTENTIAL - Must have potential to shift current understanding
            
            DETAILED EVALUATION CRITERIA:
            - NOVELTY (≥8.5/10): Explores genuinely unexplored territory, combines concepts in unprecedented ways
            - DEPTH: Requires multi-layered analysis, sophisticated methodologies, deep investigation
            - SPECIFICITY: Targets specific mechanisms/patterns, not broad generalizations
            - IMPACT: Potential for paradigm-shifting insights or revolutionary understanding
            - ORIGINALITY: Challenges existing assumptions or opens new research directions
            
            USE DEEP DOCUMENT ANALYSIS to verify genuine novelty gaps exist.
            
            FOR EACH QUESTION PROVIDE:
            - Novelty score (1-10) with harsh but fair justification
            - Depth assessment (shallow/moderate/deep) with requirements for improvement
            - Specificity evaluation (generic/focused/precise) with enhancement suggestions
            - BRUTAL HONEST FEEDBACK on why questions fail to meet world-class standards
            
            RUTHLESSLY REJECT questions scoring <8.5 novelty OR lacking depth/specificity.
            Demand revolutionary thinking, not incremental improvements.
            """,
            agent=self.agents['critic'],
            expected_output="Ruthless evaluation enforcing ≥8.5 novelty standards with detailed improvement requirements",
            context=[developer_analysis_task, supervisor_task]
        )
        
        # Task 7: Final critic review of all work
        final_critique_task = Task(
            description=f"""
            Perform final critical review of the entire research process and outputs.
            
            Evaluate:
            1. Overall research quality and coherence
            2. Adequacy of methodology (Analyzer)  
            3. Thoroughness of searches (Student)
            4. Quality of synthesis (Deep Thinker)
            5. Practical value of final recommendations
            
            Verify that the research maintains high novelty standards throughout.
            
            Provide final recommendations for improvement and assess publication readiness.
            """,
            agent=self.agents['critic'],
            expected_output="Final critical evaluation and publication readiness assessment",
            context=[novelty_critique_task, analyzer_task, student_task, philosopher_task, synthesis_task]
        )
        
        # Task 8: Reporter creates final report
        final_report_task = Task(
            description=f"""
            Create a comprehensive final report that synthesizes all research findings and addresses the original query: '{research_query}'
            
            Your report should:
            1. Directly answer the research query with high-novelty insights
            2. Present key findings from the document analysis and research
            3. Include supporting evidence from multiple sources
            4. Address criticisms and incorporate novelty feedback
            5. Provide actionable, novel conclusions
            6. Highlight the innovative aspects of the research
            
            Ensure the report reflects the high novelty standards maintained throughout the process.
            """,
            agent=self.agents['reporter'],
            expected_output="Final comprehensive research report with novel insights",
            context=[synthesis_task, final_critique_task, novelty_critique_task]
        )
        
        return [developer_analysis_task, supervisor_task, novelty_critique_task, analyzer_task, 
                student_task, philosopher_task, synthesis_task, final_critique_task, final_report_task]
    
    def conduct_research(self, research_query: str) -> Dict[str, Any]:
        """Conduct multi-agent deep research"""
        try:
            logging.info(f"Starting CrewAI deep research for: {research_query}")
            
            # Create tasks
            tasks = self._create_tasks(research_query)
            
            # Create crew with flexible process
            crew = Crew(
                agents=list(self.agents.values()),
                tasks=tasks,
                process=Process.sequential,  # Sequential but flexible based on context
                verbose=True
            )
            
            # Execute research
            result = crew.kickoff()
            
            # Format results
            formatted_result = {
                "research_query": research_query,
                "final_report": str(result),
                "agent_outputs": {},
                "metadata": {
                    "total_agents": len(self.agents),
                    "total_tasks": len(tasks),
                    "process_type": "CrewAI Multi-Agent"
                }
            }
            
            # Extract individual task outputs
            for i, task in enumerate(tasks):
                agent_name = task.agent.role.replace(' ', '_').lower()
                formatted_result["agent_outputs"][f"{i+1}_{agent_name}"] = {
                    "description": task.description[:200] + "..." if len(task.description) > 200 else task.description,
                    "output": str(task.output) if hasattr(task, 'output') else "Task output not available"
                }
            
            logging.info("CrewAI deep research completed successfully")
            return formatted_result
            
        except Exception as e:
            logging.error(f"Error in CrewAI deep research: {e}")
            return {
                "error": f"CrewAI research failed: {str(e)}",
                "research_query": research_query
            }

# Global instance
crewai_research_system = CrewAIDeepResearch()