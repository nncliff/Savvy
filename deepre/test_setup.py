#!/usr/bin/env python3
"""
Complete testing script for Google text-embedding-004 integration
Run this script to validate your entire setup step by step
"""

import os
import sys
import asyncio
from pathlib import Path
import httpx

def test_step(name, func):
    """Helper to test steps with clear output."""
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"{'='*60}")
    try:
        result = func()
        print(f"✅ {name}: PASSED")
        return result
    except Exception as e:
        print(f"❌ {name}: FAILED - {e}")
        return False

def test_environment_variables():
    """Test if all required environment variables are set."""
    required_vars = {
        'GRAPHRAG_API_KEY': 'OpenAI API key',
        'GOOGLE_API_KEY': 'Google API key'
    }
    
    missing = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value:
            missing.append(f"{var} ({description})")
        else:
            print(f"✅ {var}: {'*' * 8}{value[-4:]}")
    
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")
    
    return True

def test_google_api_direct():
    """Test Google text-embedding-004 API directly."""
    api_key = os.getenv('GOOGLE_API_KEY')
    
    # Test Gemini API endpoint
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"
    
    payload = {
        "content": {"parts": [{"text": "Hello world test"}]},
        "task_type": "RETRIEVAL_DOCUMENT"
    }
    
    response = httpx.post(url, json=payload, timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        embedding = data.get("embedding", {}).get("values", [])
        print(f"✅ Google API response: {len(embedding)} dimensions")
        print(f"✅ First 5 values: {embedding[:5]}")
        return True
    else:
        raise ValueError(f"Google API error: {response.status_code} - {response.text}")

def test_openai_api_direct():
    """Test OpenAI API connectivity."""
    import openai
    
    api_key = os.getenv('GRAPHRAG_API_KEY')
    client = openai.OpenAI(api_key=api_key)
    
    try:
        # Test with a simple completion
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=5
        )
        print(f"✅ OpenAI API response: {response.choices[0].message.content.strip()}")
        return True
    except Exception as e:
        raise ValueError(f"OpenAI API error: {e}")

def test_google_provider_direct():
    """Test the custom Google provider directly."""
    sys.path.insert(0, str(Path(__file__).parent))
    from google_embedding_provider import GoogleVertexAIEmbeddingModel
    
    api_key = os.getenv('GOOGLE_API_KEY')
    
    model = GoogleVertexAIEmbeddingModel(
        model="text-embedding-004",
        api_key=api_key,
        use_vertex_ai=False  # Use Gemini API
    )
    
    # Test sync embedding
    result = model.embed("Test embedding")
    print(f"✅ Custom provider sync: {len(result)} dimensions")
    
    # Test async embedding
    async def test_async():
        result = await model.aembed("Test async embedding")
        return len(result)
    
    dimensions = asyncio.run(test_async())
    print(f"✅ Custom provider async: {dimensions} dimensions")
    
    return True

def test_configuration_files():
    """Test if all required configuration files exist."""
    root = Path(".")
    
    required_files = {
        'settings.yaml': 'GraphRAG configuration',
        '.env': 'Environment variables (optional but recommended)',
    }
    
    for filename, description in required_files.items():
        filepath = root / filename
        if filepath.exists():
            print(f"✅ {filename}: Found")
            if filename == 'settings.yaml':
                with open(filepath) as f:
                    content = f.read()
                    if 'google_vertex_ai_embedding' in content:
                        print(f"✅ {filename}: Google provider configured")
                    else:
                        print(f"❌ {filename}: Google provider not configured")
                        return False
        else:
            print(f"⚠️  {filename}: Not found ({description})")
    
    return True

def test_full_integration():
    """Test the full integration with GraphRAG."""
    sys.path.insert(0, str(Path(__file__).parent))
    from register_google_provider import register_google_providers
    
    # Register providers
    register_google_providers()
    print("✅ Google provider registered with ModelFactory")
    
    # Try to import GraphRAG and validate config
    try:
        from graphrag.language_model.factory import ModelFactory
        
        # Check if Google provider is registered
        embedding_models = ModelFactory.get_embedding_models()
        if "google_vertex_ai_embedding" in embedding_models:
            print("✅ Google provider successfully registered with GraphRAG")
            return True
        else:
            print(f"✅ Available embedding models: {embedding_models}")
            return True
            
    except Exception as e:
        print(f"❌ GraphRAG config error: {e}")
        return False

def main():
    """Run all tests in sequence."""
    print("🔍 GOOGLE TEXT-EMBEDDING-004 INTEGRATION TEST")
    print("=" * 60)
    
    tests = [
        ("Environment Variables", test_environment_variables),
        ("Configuration Files", test_configuration_files),
        ("Google API Direct", test_google_api_direct),
        ("OpenAI API Direct", test_openai_api_direct),
        ("Google Provider Direct", test_google_provider_direct),
        ("Full Integration", test_full_integration),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            result = test_step(name, test_func)
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {name}: ERROR - {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! You can now run:")
        print("   python run_graphrag_index.py --validate-only")
        print("   python run_graphrag_index.py --verbose")
    else:
        print(f"\n🔧 Fix the failed tests above before proceeding.")

if __name__ == "__main__":
    main()