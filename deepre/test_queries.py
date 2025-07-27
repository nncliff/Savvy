#!/usr/bin/env python3
"""
Test script for GraphRAG query functionality
"""

import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from run_graphrag_query import GraphRAGQueryEngine

def test_queries():
    """Test the GraphRAG query engine with sample queries."""
    
    # Sample test queries
    test_queries = [
        {
            "query": "What are deep research agents?",
            "type": "local",
            "description": "Local search about deep research agents"
        },
        {
            "query": "What are the main components of deep research systems?",
            "type": "global", 
            "description": "Global search about DR system components"
        },
        {
            "query": "How do API-based and browser-based retrieval methods compare?",
            "type": "local",
            "description": "Local search about retrieval methods"
        }
    ]
    
    try:
        # Initialize query engine
        print("Initializing GraphRAG query engine...")
        engine = GraphRAGQueryEngine()
        print("✓ Query engine initialized successfully")
        
        # Run test queries
        for i, test in enumerate(test_queries, 1):
            print(f"\n{'='*60}")
            print(f"Test {i}: {test['description']}")
            print(f"Query: {test['query']}")
            print(f"Type: {test['type']}")
            print(f"{'='*60}")
            
            result = engine.search(test['query'], test['type'])
            
            if result['status'] == 'success':
                print("✓ Query executed successfully")
                if 'result' in result:
                    # Print a summary of the result
                    result_text = str(result['result'])
                    if len(result_text) > 500:
                        print(f"Result preview: {result_text[:500]}...")
                    else:
                        print(f"Result: {result_text}")
                else:
                    print("No result data returned")
            else:
                print(f"✗ Query failed: {result.get('error', 'Unknown error')}")
        
        print(f"\n{'='*60}")
        print("All tests completed!")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_queries()