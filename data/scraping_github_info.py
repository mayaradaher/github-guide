import requests
from bs4 import BeautifulSoup
import time
import re
import json
from urllib.parse import urljoin


BASE_URL = "https://docs.github.com"
HEADERS = {"User-Agent": "Mozilla/5.0"}


# Clean names
def clean_filename(text):
    return re.sub(r'[\\/*?:"<>|]', "_", text.strip())[:100]


# Extract the main links from the homepage
def get_main_links():
    response = requests.get(BASE_URL, headers=HEADERS, timeout=20)
    soup = BeautifulSoup(response.text, "html.parser")
    items = soup.select("li a[href]")

    links = []
    for a in items:
        href = a.get("href")
        if not href or href.startswith("#"):
            continue

        full_url = urljoin(BASE_URL, href)
        title = a.get_text(strip=True)
        if title:
            print(f"Main link: {title} -> {full_url}")
            links.append((title, full_url))

    return links


# Extract sub-links from within the pages
def extract_sub_links(url):
    try:
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")
        list_boxes = soup.select(".List__ListBox-sc-1x7olzq-0.gAwGiF")

        sublinks = []
        for box in list_boxes:
            a_tags = box.find_all("a", href=True)
            for a in a_tags:
                sub_href = a["href"]
                sub_text = a.get_text(strip=True)
                full_sub_url = (
                    sub_href if sub_href.startswith("http") else f"{BASE_URL}{sub_href}"
                )
                print(f"   ↪ Sub-link: {sub_text} -> {full_sub_url}")
                sublinks.append((sub_text, full_sub_url))
        return sublinks
    except Exception as e:
        print(f"[Error accessing {url}]: {e}")
        return []


# Extract the textual content of a URL
def extract_content(title, url, prefix=""):
    try:
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")
        main = soup.find("main")

        if not main:
            print(f"No <main> found at {url}")
            return None

        text = main.get_text(separator="\n", strip=True)
        print(f"Scraped content from: {title} ({url})")
        print(f"--- Start of content: ---\n{text[:200]}...\n--- End of preview ---\n")
        return {"section": prefix, "subtitle": title, "content": text, "source": url}
    except Exception as e:
        print(f"[Error accessing{url}]: {e}")
        return None


# Execute and save JSON file
def run_full_scraper():
    all_data = []

    main_links = get_main_links()
    for main_title, main_url in main_links:
        main_content = extract_content(main_title, main_url, prefix="main")
        if main_content:
            all_data.append(main_content)

        sublinks = extract_sub_links(main_url)
        for sub_title, sub_url in sublinks:
            content = extract_content(
                sub_title, sub_url, prefix=clean_filename(main_title)
            )
            if content:
                all_data.append(content)
            time.sleep(1)

    with open("data/scraping_github_info.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)


run_full_scraper()
