#!/usr/bin/env python3
"""
Test the Docker container using the full translation data from test_api.py
This is a modified version of test_api.py that targets the Docker container on port 8002
"""

import requests
import time
import json
import tempfile
import os
from pathlib import Path

def create_text_from_json(bookmarks_data):
    """Convert JSON bookmark data to text format expected by API"""
    
    # Start with header
    text_content = f"""Bookmark Text Content Export
Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}
Total bookmarks: {len(bookmarks_data)}
Minimum content length: 300
================================================================================

"""
    
    # Convert each bookmark to text format
    for bookmark in bookmarks_data:
        # Get translated title and content
        title = bookmark.get('title', {})
        if isinstance(title, dict):
            title_text = title.get('translated', title.get('original', 'Untitled'))
        else:
            title_text = str(title)
        
        content = bookmark.get('content', {})
        if isinstance(content, dict):
            content_text = content.get('translated', content.get('original', ''))
        else:
            content_text = str(content)
        
        # Format bookmark entry
        text_content += f"""[{bookmark.get('id', 'Unknown')}] {title_text}
URL: {bookmark.get('url', 'No URL')}
Crawled: {bookmark.get('crawled', 'Unknown')}
Content Length: {bookmark.get('content_length', 'Unknown')}
----------------------------------------
{content_text}
================================================================================

"""
    
    return text_content

def test_api_basic():
    """Test basic API functionality"""
    
    # Test health check - MODIFIED FOR DOCKER PORT 8002
    try:
        response = requests.get("http://localhost:8002/")
        print(f"Health check: {response.status_code}")
        print(f"Response: {response.json()}")
        return True
    except Exception as e:
        print(f"API not running: {e}")
        return False

def test_upload_and_process():
    """Test file upload and processing with full translation data"""
    
    # Load the actual translation file and convert to text format
    try:
        with open('translate/bookmarks_complete_translation.json', 'r', encoding='utf-8') as f:
            bookmarks_data = json.load(f)
        
        # Convert JSON to text format expected by the API
        test_content = create_text_from_json(bookmarks_data)
        print(f"Loaded {len(bookmarks_data)} bookmarks from translate/bookmarks_complete_translation.json")
        
    except FileNotFoundError:
        print("Warning: Could not find translate/bookmarks_complete_translation.json, using fallback content")
        # Fallback to original test content if file not found
        test_content = """[1] Reinforcement Learning and Kalman Filter Integration
URL: https://example.com/rl-kalman
Crawled: 2025-07-14 02:35:19.532000
Content Length: 1,860 characters
----------------------------------------
This research presents breakthrough reinforcement learning and Kalman filtering integration for autonomous systems.
"""
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        temp_file = f.name
    
    try:
        # Upload file - MODIFIED FOR DOCKER PORT 8002
        with open(temp_file, 'rb') as f:
            files = {'file': ('test_bookmark.txt', f, 'text/plain')}
            response = requests.post("http://localhost:8002/upload", files=files)
        
        if response.status_code == 200:
            result = response.json()
            job_id = result['job_id']
            print(f"Upload successful! Job ID: {job_id}")
            
            # Monitor status - MODIFIED FOR DOCKER PORT 8002
            print("Monitoring processing status...")
            for i in range(60):  # Extended timeout for full dataset
                status_response = requests.get(f"http://localhost:8002/status/{job_id}")
                if status_response.status_code == 200:
                    status = status_response.json()
                    print(f"Status: {status['status']} - {status['stage']} ({status['progress']}%)")
                    
                    if status['status'] == 'completed':
                        print("Processing completed successfully!")
                        
                        # Download report - MODIFIED FOR DOCKER PORT 8002
                        report_response = requests.get(f"http://localhost:8002/download/{job_id}")
                        if report_response.status_code == 200:
                            report_file = f"docker_full_test_report_{job_id}.md"
                            with open(report_file, 'w', encoding='utf-8') as f:
                                f.write(report_response.text)
                            print(f"Report downloaded: {report_file}")
                            
                            # Print summary
                            print(f"\n🎉 SUCCESS SUMMARY:")
                            print(f"   - Processed: {len(bookmarks_data) if 'bookmarks_data' in locals() else 'Unknown'} bookmarks")
                            print(f"   - Job ID: {job_id}")
                            print(f"   - Report: {report_file}")
                            return True
                        else:
                            print(f"Failed to download report: {report_response.status_code}")
                            return False
                    
                    elif status['status'] == 'failed':
                        print(f"Processing failed: {status['message']}")
                        return False
                    
                    time.sleep(3)  # Longer wait for full dataset
                else:
                    print(f"Failed to get status: {status_response.status_code}")
                    return False
            
            print("Processing timed out")
            return False
        else:
            print(f"Upload failed: {response.status_code} - {response.text}")
            return False
            
    finally:
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    print("🐳 Testing Docker-hosted Breakthrough Discovery API with Full Dataset...")
    print("📍 Target: http://localhost:8002/")
    
    # Test basic functionality
    if not test_api_basic():
        print("❌ Docker API is not running. Please start it with: docker compose up -d")
        exit(1)
    
    # Test upload and processing with full data
    if test_upload_and_process():
        print("✅ Docker API test with full dataset completed successfully!")
    else:
        print("❌ Docker API test failed")