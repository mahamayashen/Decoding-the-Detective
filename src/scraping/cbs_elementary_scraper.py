import requests
from bs4 import BeautifulSoup
import os
import time
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "cbs_elementary")
BASE_URL = "https://transcripts.foreverdreaming.org/viewforum.php?f=12"
DELAY_SECONDS = 3
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

def sanitize_title(raw_title):
    """Convert raw title to SxxExx Title format"""
    # Match patterns like "07x10 - The Latest Model"
    match = re.search(
        r'(\d{1,2})x(\d{1,2})\s*[-–:]?\s*(.+)', 
        raw_title,
        re.IGNORECASE
    )
    
    if match:
        season = f"{int(match.group(1)):02d}"
        episode = f"{int(match.group(2)):02d}"
        return f"S{season}E{episode} {match.group(3).strip()}"
    return raw_title.strip()

def sanitize_filename(title):
    """Clean filenames for all OSes"""
    return re.sub(r'[\\/*?:"<>|]', "", title).replace(" ", "_").strip()

def test_connectivity():
    try:
        requests.get("https://transcripts.foreverdreaming.org", timeout=5)
        return True
    except Exception as e:
        print(f"Connectivity test failed: {str(e)}")
        return False

def setup_session():
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.headers.update(HEADERS)
    return session

def scrape_episode(session, url):
    try:
        print(f"\nProcessing: {url.split('&')[0]}")  # Clean URL in logs
        response = session.get(url, timeout=20)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch episode: {str(e)}")
        return

    soup = BeautifulSoup(response.text, "html.parser")
    
    # Updated title selector based on HTML structure
    title_tag = soup.select_one('h2.topic-title a')
    if title_tag:
        raw_title = title_tag.text.strip()
        clean_title = sanitize_title(raw_title)
        title = sanitize_filename(clean_title)
    else:
        # Clean thread ID extraction
        thread_id = url.split('t=')[1].split('&')[0] if 't=' in url else "unknown"
        title = f"elementary_{thread_id}"

    content_div = soup.select_one('div.postbody div.content, div.bbcode, div.content')
    if not content_div:
        print(f"ERROR: No content found at {url}")
        return
    
    transcript = "\n".join(line.strip() for line in content_div.text.splitlines() if line.strip())
    
    filename = f"{title}.txt"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    if os.path.exists(filepath):
        print(f"Skipping duplicate: {filename}")
        return
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Source URL: {url.split('&')[0]}\n\n{transcript}")  # Clean URL in file
    print(f"Saved: {filename}")

def scrape_elementary():
    session = setup_session()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    page_num = 0
    while True:
        print(f"\n=== Scraping page {page_num + 1} ===")
        forum_url = f"{BASE_URL}&start={page_num * 25}"
        
        try:
            response = session.get(forum_url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            
            threads = soup.select('a.topictitle:not(.sticky)')
            if not threads:
                print("No more threads found. Stopping.")
                break
            
            for thread in threads:
                if (thread_url := thread.get("href")) and "viewtopic.php" in thread_url:
                    full_url = f"https://transcripts.foreverdreaming.org/{thread_url}"
                    scrape_episode(session, full_url)
                    time.sleep(DELAY_SECONDS)
            
            if not soup.select_one('li.next a'):
                break
            page_num += 1
            
        except Exception as e:
            print(f"Stopped at page {page_num}: {str(e)}")
            break

if __name__ == "__main__":
    if not test_connectivity():
        print("ERROR: Cannot reach transcripts.foreverdreaming.org")
        print("1. Check internet connection\n2. Try DNS flush\n3. Consider Google DNS (8.8.8.8)")
    else:
        print("Starting scrape...")
        scrape_elementary()
        print("\nScraping complete! Files in:", OUTPUT_DIR)