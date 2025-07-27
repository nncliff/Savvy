#!/usr/bin/env python3
"""
Programmatic GraphRAG index build with Google Vertex AI text-embedding-004
Replaces: graphrag index --root ./
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
from typing import Any

# Add current directory to Python path for custom providers
sys.path.insert(0, str(Path(__file__).parent))

# Import and register Google provider BEFORE importing graphrag
from register_google_provider import register_google_providers
register_google_providers()

# Now import graphrag components
import graphrag.api as api
from graphrag.api.index import build_index
from graphrag.config.create_graphrag_config import create_graphrag_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('graphrag_index.log')
    ]
)
logger = logging.getLogger(__name__)


class GraphRAGIndexBuilder:
    """Programmatic GraphRAG index builder with Google Vertex AI support."""
    
    def __init__(self, root_dir: str = "."):
        """Initialize the index builder."""
        self.root_dir = Path(root_dir)
        self.config_path = self.root_dir / "settings.yaml"
        self.env_file = self.root_dir / ".env"
        
        # Verify required files exist
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        # Load environment variables
        self._load_env_vars()
        
        # Register Google provider
        register_google_providers()
        logger.info("Google Vertex AI provider registered")
    
    def _load_env_vars(self):
        """Load environment variables from .env file."""
        if self.env_file.exists():
            with open(self.env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value.strip().strip('"\'')
    
    async def build_index(self, verbose: bool = True) -> dict[str, Any]:
        """Build the GraphRAG index programmatically."""
        try:
            logger.info("Starting GraphRAG index build...")
            
            # Load configuration
            from pathlib import Path
            from graphrag.config.load_config import load_config
            config = load_config(Path(str(self.root_dir)), None)
            
            # Build index using GraphRAG API
            logger.info("Building index...")
            result = await build_index(
                config=config,
                memory_profile=False,
            )
            
            # Log results
            logger.info("Index build completed successfully!")
            logger.info(f"Result: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"Index build failed: {str(e)}", exc_info=True)
            raise
    
    def print_index_summary(self, result: dict[str, Any]):
        """Print a summary of the index build results."""
        logger.info("=" * 50)
        logger.info("INDEX BUILD SUMMARY")
        logger.info("=" * 50)
        
        if 'pipeline_result' in result:
            pipeline = result['pipeline_result']
            for stage, output in pipeline.items():
                if output:
                    logger.info(f"✓ {stage}: {len(output)} records")
                else:
                    logger.info(f"✓ {stage}: Completed")
        
        # Check output files
        output_dir = self.root_dir / "output"
        if output_dir.exists():
            files = list(output_dir.rglob("*"))
            logger.info(f"Output files created: {len(files)}")
            for file in files:
                if file.is_file():
                    logger.info(f"  - {file.relative_to(output_dir)}")
    
    def validate_configuration(self) -> bool:
        """Validate the configuration before building."""
        try:
            # Check Google API configuration
            google_api_key = os.getenv("GOOGLE_API_KEY")
            
            if not google_api_key:
                logger.error("GOOGLE_API_KEY environment variable not set")
                return False
            
            # Test Google API connection
            logger.info("Testing Google Vertex AI connection...")
            from google_embedding_provider import GoogleVertexAIEmbeddingModel
            
            test_model = GoogleVertexAIEmbeddingModel(
                model="text-embedding-004",
                api_key=google_api_key,
                use_vertex_ai=False  # Use Gemini API
            )
            
            # Test with a small embedding
            result = test_model.embed("test")
            logger.info(f"✓ Google Vertex AI connection successful - embedding dimensions: {len(result)}")
            
            # Check OpenAI configuration
            openai_key = os.getenv("GRAPHRAG_API_KEY")
            if not openai_key:
                logger.error("GRAPHRAG_API_KEY environment variable not set")
                return False
            
            logger.info("✓ Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            return False


async def main():
    """Main entry point for programmatic index build."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Programmatic GraphRAG index build")
    parser.add_argument("--root", default=".", help="Root directory (default: .)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--validate-only", action="store_true", help="Only validate configuration")
    
    args = parser.parse_args()
    
    builder = GraphRAGIndexBuilder(args.root)
    
    try:
        # Validate configuration
        if not builder.validate_configuration():
            logger.error("Configuration validation failed")
            sys.exit(1)
        
        if args.validate_only:
            logger.info("Configuration validation completed successfully")
            return
        
        # Build index
        result = await builder.build_index(verbose=args.verbose)
        builder.print_index_summary(result)
        
    except Exception as e:
        logger.error(f"Index build failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())