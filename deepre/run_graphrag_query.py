#!/usr/bin/env python3
"""
GraphRAG Query Script with API Interface
Provides both CLI and web API for querying GraphRAG knowledge base
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
from typing import Any, Dict, Optional
import argparse
import json
import pandas as pd


def extract_clean_response(result):
    """Extract clean final answer from GraphRAG result."""
    if isinstance(result, str):
        return result
    elif hasattr(result, 'response'):
        return result.response
    elif isinstance(result, dict) and 'response' in result:
        return result['response']
    elif isinstance(result, tuple) and len(result) > 0:
        # GraphRAG returns (response, context_data) tuple
        return str(result[0])
    else:
        return str(result)


def make_json_serializable(obj):
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        # Handle custom objects with attributes
        return {k: make_json_serializable(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # For unknown types, convert to string
        return str(obj)

# Add current directory to Python path for custom providers
sys.path.insert(0, str(Path(__file__).parent))

# Import and register Google provider BEFORE importing graphrag
from register_google_provider import register_google_providers
register_google_providers()

# Import GraphRAG components
from graphrag.api.query import local_search, global_search
from graphrag.config.load_config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('graphrag_query.log')
    ]
)
logger = logging.getLogger(__name__)


class GraphRAGQueryEngine:
    """GraphRAG query engine with local and global search capabilities."""
    
    def __init__(self, root_dir: str = "."):
        """Initialize the query engine."""
        self.root_dir = Path(root_dir)
        self.config_path = self.root_dir / "settings.yaml"
        self.output_dir = self.root_dir / "output"
        
        # Verify required files exist
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        if not self.output_dir.exists():
            raise FileNotFoundError(f"Output directory not found: {self.output_dir}. Run index build first.")
        
        # Load environment variables
        self._load_env_vars()
        
        # Register Google provider
        register_google_providers()
        
        # Load configuration
        self.config = load_config(self.root_dir, None)
        
        # Load required data for queries
        self._load_query_data()
        
        logger.info("GraphRAG query engine initialized")
    
    def _load_query_data(self):
        """Load entities, communities, and reports for global search."""
        try:
            # Load entities
            entities_path = self.output_dir / "entities.parquet"
            if entities_path.exists():
                self.entities = pd.read_parquet(entities_path)
                logger.info(f"Loaded {len(self.entities)} entities")
            else:
                self.entities = pd.DataFrame()
                logger.warning("No entities.parquet found")
            
            # Load communities
            communities_path = self.output_dir / "communities.parquet"
            if communities_path.exists():
                self.communities = pd.read_parquet(communities_path)
                logger.info(f"Loaded {len(self.communities)} communities")
            else:
                self.communities = pd.DataFrame()
                logger.warning("No communities.parquet found")
            
            # Load community reports
            reports_path = self.output_dir / "community_reports.parquet"
            if reports_path.exists():
                self.community_reports = pd.read_parquet(reports_path)
                logger.info(f"Loaded {len(self.community_reports)} community reports")
            else:
                self.community_reports = pd.DataFrame()
                logger.warning("No community_reports.parquet found")
            
            # Load text units for local search
            text_units_path = self.output_dir / "text_units.parquet"
            if text_units_path.exists():
                self.text_units = pd.read_parquet(text_units_path)
                logger.info(f"Loaded {len(self.text_units)} text units")
            else:
                self.text_units = pd.DataFrame()
                logger.warning("No text_units.parquet found")
            
            # Load relationships for local search
            relationships_path = self.output_dir / "relationships.parquet"
            if relationships_path.exists():
                self.relationships = pd.read_parquet(relationships_path)
                logger.info(f"Loaded {len(self.relationships)} relationships")
            else:
                self.relationships = pd.DataFrame()
                logger.warning("No relationships.parquet found")
            
            # Load covariates (optional)
            covariates_path = self.output_dir / "covariates.parquet"
            if covariates_path.exists():
                self.covariates = pd.read_parquet(covariates_path)
                logger.info(f"Loaded {len(self.covariates)} covariates")
            else:
                self.covariates = None
                logger.info("No covariates.parquet found (optional)")
                
        except Exception as e:
            logger.error(f"Error loading query data: {str(e)}")
            # Initialize empty dataframes as fallback
            self.entities = pd.DataFrame()
            self.communities = pd.DataFrame()
            self.community_reports = pd.DataFrame()
            self.text_units = pd.DataFrame()
            self.relationships = pd.DataFrame()
            self.covariates = None
    
    def _load_env_vars(self):
        """Load environment variables from .env file."""
        env_file = self.root_dir / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value.strip().strip('"\'')
    
    async def local_search_query(self, query: str, clean_response: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Perform local search query.
        
        Args:
            query: The search query
            clean_response: If True, return only the final answer text
            **kwargs: Additional parameters for local search
        
        Returns:
            Dictionary containing search results and metadata
        """
        try:
            logger.info(f"Performing local search: {query}")
            
            result = await local_search(
                config=self.config,
                query=query,
                entities=self.entities,
                communities=self.communities,
                community_reports=self.community_reports,
                text_units=self.text_units,
                relationships=self.relationships,
                covariates=self.covariates,
                community_level=kwargs.get("community_level", 2),
                response_type=kwargs.get("response_type", "multiple paragraphs"),
                **{k: v for k, v in kwargs.items() if k not in ["community_level", "response_type", "clean_response"]}
            )
            
            logger.info("Local search completed successfully")
            
            # Choose response format based on clean_response flag
            if clean_response:
                response_data = extract_clean_response(result)
            else:
                response_data = make_json_serializable(result)
            
            return {
                "query": query,
                "search_type": "local",
                "result": response_data,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Local search failed: {str(e)}", exc_info=True)
            return {
                "query": query,
                "search_type": "local",
                "error": str(e),
                "status": "error"
            }
    
    async def global_search_query(self, query: str, clean_response: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Perform global search query.
        
        Args:
            query: The search query
            clean_response: If True, return only the final answer text
            **kwargs: Additional parameters for global search
        
        Returns:
            Dictionary containing search results and metadata
        """
        try:
            logger.info(f"Performing global search: {query}")
            
            # Set default parameters for global search
            search_params = {
                "entities": self.entities,
                "communities": self.communities,
                "community_reports": self.community_reports,
                "community_level": kwargs.get("community_level", 2),
                "dynamic_community_selection": kwargs.get("dynamic_community_selection", False),
                "response_type": kwargs.get("response_type", "multiple paragraphs"),
                **{k: v for k, v in kwargs.items() if k not in ["community_level", "dynamic_community_selection", "response_type", "clean_response"]}
            }
            
            result = await global_search(
                config=self.config,
                query=query,
                entities=search_params["entities"],
                communities=search_params["communities"],
                community_reports=search_params["community_reports"],
                community_level=search_params["community_level"],
                dynamic_community_selection=search_params["dynamic_community_selection"],
                response_type=search_params["response_type"],
                **{k: v for k, v in search_params.items() if k not in ["entities", "communities", "community_reports", "community_level", "dynamic_community_selection", "response_type"]}
            )
            
            logger.info("Global search completed successfully")
            
            # Choose response format based on clean_response flag
            if clean_response:
                response_data = extract_clean_response(result)
            else:
                response_data = make_json_serializable(result)
            
            return {
                "query": query,
                "search_type": "global",
                "result": response_data,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Global search failed: {str(e)}", exc_info=True)
            return {
                "query": query,
                "search_type": "global",
                "error": str(e),
                "status": "error"
            }
    
    async def search_async(self, query: str, search_type: str = "local", **kwargs) -> Dict[str, Any]:
        """
        Async search operations for use within existing event loops.
        
        Args:
            query: The search query
            search_type: Either "local" or "global"
            **kwargs: Additional search parameters
        
        Returns:
            Search results dictionary
        """
        try:
            if search_type.lower() == "local":
                return await self.local_search_query(query, **kwargs)
            elif search_type.lower() == "global":
                return await self.global_search_query(query, **kwargs)
            else:
                return {
                    "query": query,
                    "error": f"Invalid search type: {search_type}. Use 'local' or 'global'",
                    "status": "error"
                }
        except Exception as e:
            logger.error(f"Search operation failed: {str(e)}")
            return {
                "query": query,
                "error": str(e),
                "status": "error"
            }

    def search(self, query: str, search_type: str = "local", **kwargs) -> Dict[str, Any]:
        """
        Synchronous wrapper for search operations.
        
        Args:
            query: The search query
            search_type: Either "local" or "global"
            **kwargs: Additional search parameters
        
        Returns:
            Search results dictionary
        """
        try:
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # If we're in an event loop, we can't use asyncio.run()
                raise RuntimeError("Cannot use synchronous search() method within an async context. Use search_async() instead.")
            except RuntimeError:
                # No running event loop, safe to use asyncio.run()
                if search_type.lower() == "local":
                    return asyncio.run(self.local_search_query(query, **kwargs))
                elif search_type.lower() == "global":
                    return asyncio.run(self.global_search_query(query, **kwargs))
                else:
                    return {
                        "query": query,
                        "error": f"Invalid search type: {search_type}. Use 'local' or 'global'",
                        "status": "error"
                    }
        except Exception as e:
            logger.error(f"Search operation failed: {str(e)}")
            return {
                "query": query,
                "error": str(e),
                "status": "error"
            }


def cli_interface():
    """Command line interface for GraphRAG queries."""
    parser = argparse.ArgumentParser(description="GraphRAG Query Interface")
    parser.add_argument("query", help="Search query string")
    parser.add_argument("--type", choices=["local", "global"], default="local", 
                       help="Search type (default: local)")
    parser.add_argument("--root", default=".", help="Root directory (default: .)")
    parser.add_argument("--output", help="Output file for results (optional)")
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON output")
    parser.add_argument("--clean", action="store_true", help="Return only clean final answer (like CLI)")
    parser.add_argument("--full", action="store_true", help="Return full detailed response (default)")
    
    args = parser.parse_args()
    
    try:
        # Initialize query engine
        engine = GraphRAGQueryEngine(args.root)
        
        # Determine response format
        clean_response = args.clean or (not args.full and not args.output and not args.pretty)
        
        # Perform search
        result = engine.search(args.query, args.type, clean_response=clean_response)
        
        # Handle clean response differently
        if clean_response and not args.output:
            # Just print the clean answer like CLI
            if result.get('status') == 'success':
                print(result['result'])
            else:
                print(f"Error: {result.get('error', 'Unknown error')}", file=sys.stderr)
                sys.exit(1)
        else:
            # Format as JSON output
            if args.pretty:
                output = json.dumps(result, indent=2, ensure_ascii=False)
            else:
                output = json.dumps(result, ensure_ascii=False)
            
            # Save or print results
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(output)
                print(f"Results saved to: {args.output}")
            else:
                print(output)
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli_interface()