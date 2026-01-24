import requests
from bs4 import BeautifulSoup
import time
import json
from urllib.parse import urlparse, urldefrag, urljoin


# Extract links
HEADERS = {"User-Agent": "Mozilla/5.0"}


def extract_internal_links(url):
    response = requests.get(url, headers=HEADERS, timeout=20)
    soup = BeautifulSoup(response.text, "html.parser")

    links = set()
    for a in soup.select("a[href]"):
        href = a.get("href").strip()

        if not href:
            continue

        full_url = urljoin(url, href)
        full_url, _ = urldefrag(full_url)  # remove #anchor

        parsed = urlparse(full_url)

        if parsed.netloc == "docs.github.com" and parsed.path.startswith("/en/"):
            links.add(full_url)

    return links


# Extract content
def extract_content(url):
    response = requests.get(url, headers=HEADERS, timeout=20)
    soup = BeautifulSoup(response.text, "html.parser")

    main = soup.find("main")
    if not main:
        return None

    text = main.get_text(separator="\n", strip=True)
    title = soup.title.get_text(strip=True) if soup.title else url

    return {
        "title": title,
        "url": url,
        "content": text,
    }


# Complete crawler (recursive with control)
def crawl(start_url, max_pages=300):
    visited = set()
    to_visit = [start_url]
    results = []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)

        if url in visited:
            continue

        print(f"Scraping: {url}")
        visited.add(url)

        try:
            content = extract_content(url)
            if content:
                results.append(content)

            links = extract_internal_links(url)
            for link in links:
                if link not in visited:
                    to_visit.append(link)

            time.sleep(1)

        except Exception as e:
            print(f"Erro em {url}: {e}")

    return results


# Execute and save
data = crawl("https://docs.github.com/en/get-started", max_pages=500)

with open("data/scraper/scraping_github_info.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
