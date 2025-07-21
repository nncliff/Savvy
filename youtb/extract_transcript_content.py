import asyncio
from playwright.async_api import async_playwright
import json
import re

async def extract_full_transcript(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        try:
            print(f"Loading YouTube video: {url}")
            await page.goto(url, timeout=30000)
            await page.wait_for_load_state('networkidle')
            await page.wait_for_timeout(3000)
            
            video_data = {}
            
            # Get basic info
            title = await page.title()
            video_data['title'] = title.replace(' - YouTube', '').strip()
            
            # Extract description by clicking "Show more"
            try:
                show_more = await page.query_selector('tp-yt-paper-button#expand')
                if show_more and await show_more.is_visible():
                    await show_more.click()
                    await page.wait_for_timeout(2000)
                
                description_elem = await page.query_selector('#description-inline-expander')
                if description_elem:
                    description = await description_elem.inner_text()
                    video_data['description'] = description.strip()
                else:
                    video_data['description'] = "Description not found"
                    
            except Exception as e:
                video_data['description'] = f"Error: {e}"
            
            # Extract key moments from description
            description = video_data.get('description', '')
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
            
            video_data['key_moments'] = key_moments
            
            # Now focus on extracting the actual transcript
            transcript_lines = []
            
            print("Looking for transcript button...")
            
            # Look for "Show transcript" button or similar
            transcript_button_found = False
            
            # Try multiple approaches to find transcript button
            approaches = [
                # Look for "Show transcript" link in description
                lambda: page.query_selector('a[href*="transcript"]'),
                # Look for transcript button
                lambda: page.query_selector('button[aria-label*="transcript" i]'),
                # Look for text containing "Show transcript"
                lambda: page.locator('text="Show transcript"'),
                # Look for "Transcript" text
                lambda: page.locator('text="Transcript"'),
                # Look in more actions menu
                lambda: None  # Will try menu approach below
            ]
            
            transcript_button = None
            for approach in approaches:
                try:
                    transcript_button = await approach()
                    if transcript_button:
                        print(f"Found transcript button using approach")
                        transcript_button_found = True
                        break
                except:
                    continue
            
            # If no button found, try the three dots menu
            if not transcript_button_found:
                print("Trying three dots menu...")
                try:
                    # Look for more actions button
                    more_button = await page.query_selector('button[aria-label*="More actions"]')
                    if more_button:
                        await more_button.click()
                        await page.wait_for_timeout(2000)
                        
                        # Look for transcript in dropdown
                        transcript_option = await page.query_selector('tp-yt-paper-item:has-text("transcript" i)')
                        if transcript_option:
                            transcript_button = transcript_option
                            transcript_button_found = True
                except Exception as e:
                    print(f"Error with dropdown: {e}")
            
            if transcript_button:
                print("Clicking transcript button...")
                try:
                    await transcript_button.click()
                    await page.wait_for_timeout(3000)
                    
                    # Look for the transcript popup/box
                    print("Waiting for transcript box to appear...")
                    
                    # Wait for the transcript container to load
                    await page.wait_for_selector('[class*="transcript"], [role="dialog"], yt-structured-description-transcript-renderer', timeout=10000)
                    
                    # Look for the transcript box with title "Transcript"
                    transcript_container = None
                    container_selectors = [
                        'yt-structured-description-transcript-renderer',
                        '[class*="transcript"]',
                        '[aria-label="Transcript"]',
                        'h3:has-text("Transcript") + *',
                        '[role="dialog"] h3:has-text("Transcript")'
                    ]
                    
                    for selector in container_selectors:
                        try:
                            transcript_container = await page.query_selector(selector)
                            if transcript_container:
                                print(f"Found transcript container: {selector}")
                                break
                        except:
                            continue
                    
                    if transcript_container:
                        # Look for the actual transcript text elements
                        text_selectors = [
                            '[class*="transcript-line"]',
                            '[class*="segment"]',
                            '[class*="cue"]',
                            '[class*="text"]',
                            'yt-formatted-string[class*="transcript"]'
                        ]
                        
                        transcript_lines = []
                        
                        # Try each selector
                        for selector in text_selectors:
                            try:
                                text_elements = await page.query_selector_all(selector)
                                if text_elements:
                                    print(f"Found {len(text_elements)} transcript lines using {selector}")
                                    for element in text_elements:
                                        text = await element.text_content()
                                        if text and text.strip():
                                            # Clean up the text
                                            clean_text = text.strip()
                                            if clean_text and not clean_text.startswith('(') and len(clean_text) > 5:
                                                transcript_lines.append(clean_text)
                                    
                                    if transcript_lines:
                                        break  # Found transcript successfully
                            except Exception as e:
                                continue
                        
                        video_data['transcript_lines'] = transcript_lines
                        
                        # Also try to get timestamped transcript
                        timestamped_lines = []
                        try:
                            # Look for elements that contain both timestamp and text
                            line_elements = await page.query_selector_all('[class*="transcript-line"], [class*="segment"]')
                            
                            for line in line_elements:
                                try:
                                    # Find timestamp
                                    time_elem = await line.query_selector('[class*="time"], [class*="timestamp"]')
                                    time_text = await time_elem.text_content() if time_elem else ""
                                    
                                    # Find text
                                    text_elem = await line.query_selector('[class*="text"], [class*="content"]')
                                    text_content = await text_elem.text_content() if text_elem else ""
                                    
                                    if time_text and text_content:
                                        timestamped_lines.append({
                                            'timestamp': time_text.strip(),
                                            'text': text_content.strip()
                                        })
                                except:
                                    continue
                            
                            video_data['transcript_with_timestamps'] = timestamped_lines
                            
                        except Exception as e:
                            print(f"Error getting timestamped transcript: {e}")
                    
                    else:
                        print("Transcript container not found")
                        video_data['transcript_lines'] = ["Transcript container found but content not accessible"]
                        
                except Exception as e:
                    print(f"Error clicking transcript button: {e}")
                    video_data['transcript_lines'] = [f"Error accessing transcript: {str(e)}"]
            else:
                print("Transcript button not found")
                video_data['transcript_lines'] = ["Transcript button not accessible"]
            
            return video_data
            
        except Exception as e:
            return {'error': str(e)}
        finally:
            await browser.close()

async def main():
    url = 'https://www.youtube.com/watch?v=LMbmiJqntbE'
    video_info = await extract_full_transcript(url)
    
    # Save to JSON file
    with open('full_transcript_info.json', 'w', encoding='utf-8') as f:
        json.dump(video_info, f, indent=2, ensure_ascii=False)
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("COMPLETE VIDEO WITH FULL TRANSCRIPT")
    print("="*80)
    print(f"Title: {video_info.get('title', 'N/A')}")
    print(f"Transcript Lines: {len(video_info.get('transcript_lines', []))}")
    print(f"Timestamped Lines: {len(video_info.get('transcript_with_timestamps', []))}")
    print(f"Key Moments: {len(video_info.get('key_moments', []))}")
    
    print(f"\nKey Moments/Chapters:")
    for i, moment in enumerate(video_info.get('key_moments', []), 1):
        print(f"  {i}. {moment['timestamp']} - {moment['title']}")
    
    print(f"\nTranscript Preview (first 10 lines):")
    for i, line in enumerate(video_info.get('transcript_lines', [])[:10], 1):
        print(f"  {i}. {line}")
    
    if video_info.get('transcript_with_timestamps'):
        print(f"\nTimestamped Transcript Preview:")
        for i, entry in enumerate(video_info.get('transcript_with_timestamps', [])[:5], 1):
            print(f"  {i}. [{entry['timestamp']}] {entry['text']}")
    
    print(f"\nFull details saved to: full_transcript_info.json")

if __name__ == "__main__":
    asyncio.run(main())