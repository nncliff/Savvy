#!/usr/bin/env python3
"""
Test script for the deep research functionality.
This script tests the Plan → Think → Action → Analyze pattern implementation.
"""

import requests
import json
import time
from dotenv import load_dotenv

load_dotenv()

# Configuration
BASE_URL = "http://localhost:8000"

def test_api_health():
    """Test if the API is running and healthy"""
    print("🔍 Testing API health...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✓ API is running")
            return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"❌ Failed to connect to API: {e}")
        return False

def test_index_status():
    """Test the index status"""
    print("\n🔍 Testing index status...")
    try:
        response = requests.get(f"{BASE_URL}/index-status")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Index ready: {data.get('index_ready', False)}")
            print(f"✓ Total indexed bookmarks: {data.get('total_indexed_bookmarks', 0)}")
            print(f"✓ Documents in memory: {data.get('documents_in_memory', 0)}")
            return data.get('index_ready', False)
        else:
            print(f"❌ Index status check failed: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"❌ Failed to check index status: {e}")
        return False

def test_simple_query():
    """Test the basic query functionality"""
    print("\n🔍 Testing simple query...")
    try:
        test_query = "artificial intelligence"
        response = requests.post(
            f"{BASE_URL}/query",
            json={"query": test_query}
        )
        if response.status_code == 200:
            data = response.json()
            if "error" in data:
                print(f"❌ Query returned error: {data['error']}")
                return False
            else:
                print(f"✓ Simple query successful")
                print(f"  Response preview: {str(data.get('response', ''))[:100]}...")
                return True
        else:
            print(f"❌ Query failed with status: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"❌ Query request failed: {e}")
        return False

def test_deep_research():
    """Test the deep research functionality"""
    print("\n🔍 Testing deep research functionality...")
    
    research_queries = [
        "machine learning trends",
        "sustainable technology",
        "remote work productivity"
    ]
    
    for query in research_queries:
        print(f"\n📚 Testing deep research for: '{query}'")
        try:
            response = requests.post(
                f"{BASE_URL}/deep-research",
                json={"research_query": query}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if "error" in data:
                    print(f"❌ Deep research returned error: {data['error']}")
                    continue
                
                # Verify the structure of the response
                required_keys = ["research_query", "plan", "findings", "analysis"]
                missing_keys = [key for key in required_keys if key not in data]
                
                if missing_keys:
                    print(f"❌ Missing keys in response: {missing_keys}")
                    continue
                
                print("✓ Deep research completed successfully")
                print(f"  Research Query: {data['research_query']}")
                
                # Check plan structure
                plan = data.get('plan', {})
                if 'research_topic' in plan and 'plan' in plan:
                    print(f"  Plan generated with {len(plan['plan'])} questions")
                else:
                    print("❌ Invalid plan structure")
                    continue
                
                # Check findings structure
                findings = data.get('findings', [])
                print(f"  Findings collected for {len(findings)} questions")
                
                for i, finding in enumerate(findings):
                    if all(key in finding for key in ['question', 'reasoning', 'search_queries', 'search_results']):
                        search_count = len(finding['search_results'])
                        print(f"    Question {i+1}: {search_count} searches performed")
                    else:
                        print(f"    Question {i+1}: Invalid structure")
                
                # Check analysis
                analysis = data.get('analysis', '')
                if analysis and len(analysis) > 100:
                    print(f"  Analysis generated ({len(analysis)} characters)")
                    print(f"  Analysis preview: {analysis[:150]}...")
                else:
                    print("❌ Analysis too short or missing")
                
                print("✓ Deep research structure validation passed")
                
            else:
                print(f"❌ Deep research failed with status: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error details: {error_data}")
                except:
                    print(f"   Raw response: {response.text}")
                    
        except requests.RequestException as e:
            print(f"❌ Deep research request failed: {e}")
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse response JSON: {e}")
        
        print()  # Space between queries

def test_deep_research_get():
    """Test the GET endpoint for deep research"""
    print("\n🔍 Testing deep research GET endpoint...")
    try:
        test_query = "productivity tools"
        response = requests.get(
            f"{BASE_URL}/deep-research",
            params={"research_query": test_query}
        )
        
        if response.status_code == 200:
            data = response.json()
            if "error" in data:
                print(f"❌ GET endpoint returned error: {data['error']}")
                return False
            else:
                print("✓ Deep research GET endpoint working")
                return True
        else:
            print(f"❌ GET endpoint failed with status: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"❌ GET endpoint request failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests"""
    print("🧪 Deep Research Functionality Test Suite")
    print("=" * 50)
    
    # Test sequence
    tests = [
        ("API Health", test_api_health),
        ("Index Status", test_index_status),
        ("Simple Query", test_simple_query),
        ("Deep Research POST", test_deep_research),
        ("Deep Research GET", test_deep_research_get)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print("-" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Deep research functionality is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")
    
    return passed == total

if __name__ == "__main__":
    print("🚀 Starting Deep Research Test Suite...")
    print("Make sure the RAG API is running on http://localhost:8000")
    print("You can start it with: uvicorn rag_api:app --host 0.0.0.0 --port 8000")
    print()
    
    input("Press Enter to start tests...")
    success = run_comprehensive_test()
    
    if success:
        exit(0)
    else:
        exit(1)