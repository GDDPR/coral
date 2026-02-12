import hashlib
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

INDEX_URL = (
    "https://www.canada.ca/en/department-national-defence/corporate/policies-standards/"
    "defence-administrative-orders-directives/6000-series.html"
)


def url_hash_id(u: str) -> str:
    """
    Dedupe-by-URL-hash id (stable).
    """
    return hashlib.sha256(u.encode("utf-8")).hexdigest()[:16]


def extract_state_from_inner_li(inner_li) -> str | None:
    """
    State is inside <strong><em>...</em></strong> within the inner <li>.
    Text looks like "(under review)" -> return "under review".
    """
    node = inner_li.select_one("strong em") or inner_li.select_one("em strong")
    if not node:
        return None

    text = node.get_text(strip=True)
    if not text:
        return None

    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()

    return text or None


def fetch_index_html(url: str = INDEX_URL, timeout: int = 20) -> str:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def find_items_from_index_html(html: str, base_url: str = INDEX_URL) -> list[dict]:
    """
    Returns items like:
      {"url": "...", "state": "under review" or None}

    (No timestamps/status/domain here — this file is purely scraping.)
    """
    soup = BeautifulSoup(html, "lxml")

    outer_ul = soup.find("ul", id="w1384333708783")
    if not outer_ul:
        return []

    seen: set[str] = set()
    items: list[dict] = []

    for top_li in outer_ul.find_all("li", recursive=False):
        for inner_ul in top_li.find_all("ul"):
            for inner_li in inner_ul.find_all("li", recursive=False):
                a = inner_li.find("a", href=True)
                if not a:
                    continue

                href = a["href"].strip()
                if not href:
                    continue

                full_url = urljoin(base_url, href)

                if full_url in seen:
                    continue
                seen.add(full_url)

                state = extract_state_from_inner_li(inner_li)

                items.append(
                    {
                        "url": full_url,
                        "state": state,
                    }
                )

    return items


def scrape_daod6000_items(index_url: str = INDEX_URL) -> list[dict]:
    html = fetch_index_html(index_url)
    return find_items_from_index_html(html, base_url=index_url)


if __name__ == "__main__":
    items = scrape_daod6000_items()
    print(f"Found {len(items)} items.")
    for it in items[:10]:
        print(it["url"], "| state:", it["state"])
