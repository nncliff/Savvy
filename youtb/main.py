from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import asyncio
from playwright.async_api import async_playwright
import re
import logging
from typing import Optional, List, Dict, Any
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YouTube Data Extractor", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class YouTubeResponse(BaseModel):
    title: str
    description: str
    views: str
    upload_date: str
    channel: str
    comment_count: str
    transcript_available: bool
    key_moments: List[Dict[str, str]]
    transcript_lines: List[str]
    transcript_with_timestamps: List[Dict[str, str]]
    url: str
    status: str

class ErrorResponse(BaseModel):
    error: str
    message: str
    url: str

def validate_youtube_url(url: str) -> bool:
    """Validate if URL is a valid YouTube URL"""
    youtube_patterns = [
        r'^https?://(?:www\.)?youtube\.com/watch\?v=([\w-]+)',
        r'^https?://(?:www\.)?youtu\.be/([\w-]+)',
        r'^https?://(?:www\.)?youtube\.com/embed/([\w-]+)',
    ]
    
    for pattern in youtube_patterns:
        if re.match(pattern, url):
            return True
    return False

async def extract_youtube_data(url: str) -> Dict[str, Any]:
    """Extract YouTube data using Playwright"""
    if not validate_youtube_url(url):
        raise ValueError("Invalid YouTube URL format")
    
    async with async_playwright() as p:
        browser = None
        try:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # Set reasonable timeout
            page.set_default_timeout(30000)
            
            logger.info(f"Loading YouTube video: {url}")
            await page.goto(url, timeout=30000)
            await page.wait_for_load_state('networkidle')
            await page.wait_for_timeout(3000)
            
            data = {
                'url': url,
                'status': 'success'
            }
            
            # Extract title
            title = await page.title()
            data['title'] = title.replace(' - YouTube', '').strip()
            
            # Extract description
            try:
                show_more = await page.query_selector('tp-yt-paper-button#expand')
                if show_more and await show_more.is_visible():
                    await show_more.click()
                    await page.wait_for_timeout(2000)
                
                description_elem = await page.query_selector('#description-inline-expander')
                if description_elem:
                    description = await description_elem.inner_text()
                    data['description'] = description.strip()
                else:
                    data['description'] = "Description not available"
                    
            except Exception as e:
                data['description'] = f"Error extracting description: {str(e)}"
            
            # Extract key moments
            description = data.get('description', '')
            key_moments = []
            timestamp_pattern = r'(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–—]\s*([^\n]+)'
            matches = re.findall(timestamp_pattern, description)
            
            for timestamp, title in matches:
                title = title.strip()
                if title and len(title) > 3 and not title.startswith('http'):
                    key_moments.append({
                        'timestamp': timestamp.strip(),
                        'title': title
                    })
            
            data['key_moments'] = key_moments
            
            # Extract metadata
            content = await page.content()
            
            # Views
            views_match = re.search(r'(\d+(?:,\d{3})*(?:,\d{3})?\s*views)', content, re.IGNORECASE)
            if not views_match:
                views_match = re.search(r'(\d+(?:\.\d+)?[KMB]?\s*views)', content, re.IGNORECASE)
            data['views'] = views_match.group(1) if views_match else "Views not found"
            
            # Upload date
            date_match = re.search(r'(\w+\s+\d{1,2},\s+\d{4})', content)
            data['upload_date'] = date_match.group(1) if date_match else "Date not found"
            
            # Channel
            channel_match = re.search(r'"ownerChannelName":"([^"]+)"', content)
            if channel_match:
                data['channel'] = channel_match.group(1)
            else:
                channel_elem = await page.query_selector('#channel-name a')
                if channel_elem:
                    channel_text = await channel_elem.text_content()
                    data['channel'] = channel_text.strip()
                else:
                    data['channel'] = "Channel not found"
            
            # Comment count
            comment_match = re.search(r'(\d+(?:,\d{3})*(?:,\d{3})?\s*Comments)', content, re.IGNORECASE)
            data['comment_count'] = comment_match.group(1) if comment_match else "Comments not found"
            
            # Check transcript availability
            data['transcript_available'] = "Show transcript" in content
            
            # Extract transcript
            transcript_lines = []
            transcript_timestamps = []
            
            if data['transcript_available']:
                try:
                    logger.info("Attempting to extract transcript...")
                    
                    # First, try to find transcript button directly
                    transcript_button = None
                    direct_selectors = [
                        'button[aria-label*="Show transcript"]',
                        'button[title*="Show transcript"]',
                        'button:has-text("Show transcript")',
                        'text="Show transcript"'
                    ]
                    
                    # Try direct transcript button first
                    for selector in direct_selectors:
                        try:
                            if selector.startswith('text='):
                                transcript_button = await page.locator(selector).first
                            else:
                                transcript_button = await page.query_selector(selector)
                            if transcript_button and await transcript_button.is_visible():
                                logger.info("Found direct transcript button")
                                await transcript_button.click()
                                await page.wait_for_timeout(4000)
                                break
                        except:
                            continue
                    
                    # If no direct button found, look for more menu
                    if not transcript_button:
                        logger.info("Looking for more menu (three dots)...")
                        more_selectors = [
                            'button[aria-label*="More actions"]',
                            'button[aria-label*="more"]',
                            'button[title*="more"]',
                            'button[data-tooltip*="more"]',
                            'button[aria-label="More actions"]',
                            'button[aria-label="Action menu"]',
                            'ytd-menu-renderer button',
                            '[aria-label="More actions"]'
                        ]
                        
                        more_button = None
                        for selector in more_selectors:
                            try:
                                more_button = await page.query_selector(selector)
                                if more_button and await more_button.is_visible():
                                    logger.info(f"Found more menu with selector: {selector}")
                                    await more_button.click()
                                    await page.wait_for_timeout(3000)
                                    break
                            except:
                                continue
                        
                        # Look for "Show transcript" in the expanded menu
                        if more_button:
                            logger.info("Looking for transcript in expanded menu...")
                            transcript_selectors = [
                                'text="Show transcript"',
                                'text="Transcript"',
                                'text="Open transcript"',
                                'tp-yt-paper-item:has-text("transcript")',
                                'yt-formatted-string:has-text("transcript")',
                                '[role="menuitem"]:has-text("transcript")',
                                'button:has-text("transcript")'
                            ]
                            
                            for selector in transcript_selectors:
                                try:
                                    if selector.startswith('text='):
                                        show_transcript = await page.locator(selector).first
                                    else:
                                        show_transcript = await page.query_selector(selector)
                                    
                                    if show_transcript and await show_transcript.is_visible():
                                        logger.info(f"Found transcript option: {selector}")
                                        await show_transcript.click()
                                        await page.wait_for_timeout(4000)
                                        break
                                except Exception as e:
                                    logger.debug(f"Failed to find transcript with {selector}: {e}")
                                    continue
                    
                    if show_transcript:
                        await show_transcript.click()
                        await page.wait_for_timeout(4000)
                        
                        # Look for the transcript panel
                        transcript_panel = None
                        panel_selectors = [
                            '[aria-label="Transcript"]',
                            'ytd-transcript-renderer',
                            '[class*="transcript-panel"]',
                            '[class*="transcript-body"]'
                        ]
                        
                        for selector in panel_selectors:
                            transcript_panel = await page.query_selector(selector)
                            if transcript_panel:
                                break
                        
                        if transcript_panel:
                            # Wait for content to load
                            await page.wait_for_timeout(2000)
                            
                            # Scroll to load all transcript segments
                            await transcript_panel.scroll_into_view_if_needed()
                            await page.wait_for_timeout(1000)
                            
                            # Extract timestamped transcript
                            timestamp_pattern = re.compile(r'(\d{1,2}:\d{2}(?::\d{2})?)')
                            
                            # Get all transcript segments
                            segment_selectors = [
                                '[class*="segment"]',
                                '[class*="cue"]',
                                'ytd-transcript-segment-renderer',
                                '[class*="transcript-line"]'
                            ]
                            
                            for selector in segment_selectors:
                                segments = await transcript_panel.query_selector_all(selector)
                                if segments:
                                    for segment in segments:
                                        try:
                                            # Get timestamp
                                            time_elem = await segment.query_selector('[class*="time"], [class*="timestamp"]')
                                            timestamp = ""
                                            if time_elem:
                                                timestamp = (await time_elem.text_content()).strip()
                                            
                                            # Get text
                                            text_elem = await segment.query_selector('[class*="text"], [class*="content"], span')
                                            text = ""
                                            if text_elem:
                                                text = (await text_elem.text_content()).strip()
                                            
                                            if text and text.strip():
                                                clean_text = text.strip()
                                                transcript_lines.append(clean_text)
                                                
                                                if timestamp and timestamp_pattern.match(timestamp):
                                                    transcript_timestamps.append({
                                                        'timestamp': timestamp,
                                                        'text': clean_text
                                                    })
                                                else:
                                                    # Try to extract timestamp from text
                                                    match = timestamp_pattern.search(text)
                                                    if match:
                                                        transcript_timestamps.append({
                                                            'timestamp': match.group(1),
                                                            'text': clean_text
                                                        })
                                                    else:
                                                        transcript_timestamps.append({
                                                            'timestamp': "",
                                                            'text': clean_text
                                                        })
                                                        
                                        except Exception as e:
                                            continue
                                    
                                    if transcript_lines:
                                        break
                            
                            # If no structured segments found, try to get raw text
                            if not transcript_lines:
                                all_text = await transcript_panel.text_content()
                                if all_text:
                                    lines = [line.strip() for line in all_text.split('\n') if line.strip()]
                                    
                                    # Parse timestamped lines
                                    for line in lines:
                                        match = re.match(r'(\d{1,2}:\d{2}(?::\d{2})?)\s*(.+)', line)
                                        if match:
                                            timestamp, text = match.groups()
                                            transcript_timestamps.append({
                                                'timestamp': timestamp,
                                                'text': text.strip()
                                            })
                                            transcript_lines.append(text.strip())
                                        elif line.strip() and not re.match(r'^\d+$', line.strip()):
                                            transcript_lines.append(line.strip())
                
                except Exception as e:
                    transcript_lines = [f"Transcript extraction error: {str(e)}"]
                    transcript_timestamps = []
            
            data['transcript_lines'] = transcript_lines
            data['transcript_with_timestamps'] = transcript_timestamps
            
            return data
            
        except Exception as e:
            if browser:
                await browser.close()
            raise e

@app.get("/", response_model=YouTubeResponse)
async def extract_youtube_data_endpoint(
    url: str = Query(..., description="YouTube video URL")
):
    """Extract YouTube video data"""
    try:
        data = await extract_youtube_data(url)
        return YouTubeResponse(**data)
    
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid URL", "message": str(e), "url": url}
        )
    
    except Exception as e:
        logger.error(f"Error processing URL {url}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Processing Error", "message": str(e), "url": url}
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "youtube-extractor"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)