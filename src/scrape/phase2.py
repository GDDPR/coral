import json
import os
import re
import time
import hashlib
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
from lxml import etree

from catalog_utils import update_status


CATALOG_PATH = "./data/catalog.xml"
OUT_JSONL = "./data/docs.jsonl"  # kept (unused) to minimize changes
OUT_JSON_DIR = "./data/docs_json"  # pretty JSON files (one per link)
CACHE_DIR = "./data/cache_html"  # kept (unused) to minimize changes


def _h2_is_authorities_or_responsibilities(title: str) -> bool:
    # Handles: "3. Authorities", "2 Responsibilities", "4.1   Authorities", etc.
    return re.search(r"(authorities|responsibilities)\s*$", title.strip(), re.I) is not None


def now_iso() -> str:
    # match "...Z" style
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def stable_id_from_url(u: str) -> str:
    # Stable ID for Neo4j nodes (use canonical_url if you have it)
    return sha256_text(u)[:16]


def safe_filename(name: str) -> str:
    # keep filenames OS-safe
    name = name.strip()
    name = re.sub(r"[^\w\-\.]+", "_", name)  # letters/numbers/_/-/.
    return name[:120] or "doc"


def write_pretty_json(out_dir: str, doc: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)

    # one file per link, stable name from id
    filename = f"{safe_filename(doc['id'])}.json"
    path = os.path.join(out_dir, filename)

    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    os.replace(tmp_path, path)
    return path


# reads your XML catalog file and turns each <Item> into a Python dictionary
def load_catalog_items(xml_path: str) -> list[dict]:
    tree = etree.parse(xml_path)
    root = tree.getroot()

    items = []
    for item in root.findall("Item"):
        url_el = item.find("url")
        status_el = item.find("status")
        state_el = item.find("state")

        url = (url_el.text or "").strip() if url_el is not None else ""
        status = (status_el.text or "").strip() if status_el is not None else ""
        state = (state_el.text or "").strip() if state_el is not None else ""

        if url:
            items.append(
                {
                    "url": url,
                    "status": status,
                    "state": state,
                }
            )
    return items


# Fetch HTML with retries + exponential backoff (no caching)
def fetch_html(url: str, session: requests.Session) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (Phase2Parser)"}
    last_err: Exception | None = None

    for attempt in range(1, 5):  # 4 tries
        try:
            resp = session.get(url, headers=headers, timeout=25)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            last_err = e
            sleep_s = 1.5 * (2 ** (attempt - 1))
            time.sleep(sleep_s)
    raise last_err  # type: ignore


def extract_canonical_url(soup: BeautifulSoup, page_url: str) -> str:
    link = soup.find("link", rel=lambda v: v and "canonical" in v.lower())
    if link and link.get("href"):
        return urljoin(page_url, link["href"].strip())
    return page_url


# Find meta tags
def first_meta(soup: BeautifulSoup, *, name: str | None = None, prop: str | None = None) -> str | None:
    if name:
        tag = soup.find("meta", attrs={"name": name})
        if tag and tag.get("content"):
            return tag["content"].strip()
    if prop:
        tag = soup.find("meta", attrs={"property": prop})
        if tag and tag.get("content"):
            return tag["content"].strip()
    return None


# Extract meta tags info
def extract_title(soup: BeautifulSoup) -> str | None:
    return first_meta(soup, name="dcterms.title")


def extract_author(soup: BeautifulSoup) -> str | None:
    return first_meta(soup, name="author")


def extract_dates(soup: BeautifulSoup) -> tuple[str | None, str | None]:
    # published: issued
    published_date = first_meta(soup, name="dcterms.issued")
    # effective/modified: modified
    effective_date = first_meta(soup, name="dcterms.modified")
    return effective_date, published_date


def extract_language(soup: BeautifulSoup) -> str | None:
    lang = first_meta(soup, name="dcterms.language")
    if lang:
        return lang
    # fallback: <html lang="en">
    html = soup.find("html")
    if html and html.get("lang"):
        return str(html.get("lang")).strip()
    return None


def extract_subjects(soup: BeautifulSoup) -> list[str]:
    # Find the TOC heading
    h2 = soup.find("h2", string=re.compile(r"^\s*Table of Contents\s*$", re.I))
    if not h2:
        return []

    # The TOC list is usually inside the same parent <div>
    container = h2.find_parent("div")
    if not container:
        return []

    ol = container.find("ol")
    if not ol:
        return []

    # Subjects = text of each <li> in that <ol>
    return [li.get_text(" ", strip=True) for li in ol.find_all("li") if li.get_text(strip=True)]


def _is_within_main(tag: Tag, main: Tag) -> bool:
    cur = tag
    while cur is not None:
        if cur is main:
            return True
        cur = cur.parent  # type: ignore[assignment]
    return False


def _strip_trailing_ellipsis(s: str) -> str:
    # remove trailing "..." or "…"
    s = s.strip()
    s = re.sub(r"\s*(\.\.\.|…)\s*$", "", s).strip()
    return s


def _cell_text(cell: Tag) -> str:
    return cell.get_text(" ", strip=True)


def _get_table_header_and_row_cells(table_tag: Tag) -> tuple[list[str], list[tuple[Tag, Tag]]]:
    """
    Returns:
      header_cells_texts (first two cells, ellipsis stripped)
      data_rows as list of (left_cell_tag, right_cell_tag)

    IMPORTANT behavior:
    - Always treat the *first row* as the header row (whether it's in <thead> or not),
      and do NOT return it in data_rows.
    - Some pages have <thead> but STILL repeat the header row as the first row in <tbody>.
      In that case, we detect and skip that duplicate too.
    """
    thead = table_tag.find("thead")
    tbody = table_tag.find("tbody")

    header_tr = None
    if thead:
        header_tr = thead.find("tr")
    if header_tr is None:
        header_tr = table_tag.find("tr")

    header_cells: list[str] = []
    if header_tr:
        hdr_cells = header_tr.find_all(["th", "td"], recursive=False)
        if not hdr_cells:
            hdr_cells = header_tr.find_all(["th", "td"])
        header_cells = [_strip_trailing_ellipsis(_cell_text(c)) for c in hdr_cells[:2]]

    # Determine candidate data trs
    if thead:
        if tbody:
            trs = tbody.find_all("tr")
        else:
            trs = table_tag.find_all("tr")
            if header_tr in trs:
                trs = trs[trs.index(header_tr) + 1 :]
    else:
        trs = table_tag.find_all("tr")
        trs = trs[1:] if trs else []

    # If the first "data" row is actually a duplicated header row (common),
    # skip it:
    if trs:
        first = trs[0]
        first_cells = first.find_all(["th", "td"])
        if first_cells:
            t1 = _strip_trailing_ellipsis(_cell_text(first_cells[0])) if len(first_cells) >= 1 else ""
            t2 = _strip_trailing_ellipsis(_cell_text(first_cells[1])) if len(first_cells) >= 2 else ""
            # Heuristics: has <th> OR matches the header text
            if first.find("th") is not None or (
                len(header_cells) >= 2 and t1 == header_cells[0] and t2 == header_cells[1]
            ):
                trs = trs[1:]

    data_rows: list[tuple[Tag, Tag]] = []
    for tr in trs:
        cells = tr.find_all("td", recursive=False)
        if len(cells) < 2:
            cells = tr.find_all(["td", "th"], recursive=False)
        if len(cells) < 2:
            cells = tr.find_all(["td", "th"])
        if len(cells) < 2:
            continue
        data_rows.append((cells[0], cells[1]))

    return header_cells, data_rows


def _format_table_inline(table_tag: Tag) -> list[str]:
    """
    Non-special 2-column tables:

    Each data row becomes one paragraph:
      <header1> <row_left> <header2> <row_right bullets/text>

    IMPORTANT:
    - We do NOT output the first/header row as a data row.
    - We do not output literal newlines in JSON strings.
    """
    headers, rows = _get_table_header_and_row_cells(table_tag)
    if len(headers) < 2 or not rows:
        return []

    h1 = headers[0].strip()
    h2 = headers[1].strip()

    out: list[str] = []
    for left_cell, right_cell in rows:
        left = _cell_text(left_cell).strip()

        lis = right_cell.find_all("li")
        if lis:
            bullets = [li.get_text(" ", strip=True).strip() for li in lis if li.get_text(strip=True)]
            right_text = " ".join(bullets).strip()
        else:
            right_text = _cell_text(right_cell).strip()

        row_header = f"{h1} {left} {h2}".strip()
        paragraph = f"{row_header} {right_text}".strip() if right_text else row_header

        if paragraph:
            out.append(paragraph)

    return out


def _parse_table_authorities_responsibilities(table_tag: Tag) -> list[dict]:
    """
    For h2 ending in Authorities/Responsibilities ONLY:
    - skip header row
    - one {title,text} per data row
    - prepend title to text
    """
    headers, rows = _get_table_header_and_row_cells(table_tag)
    header2 = headers[1].strip() if len(headers) >= 2 else ""

    out: list[dict] = []
    for left_cell, right_cell in rows:
        title = _cell_text(left_cell).strip()
        if not title:
            continue

        lis = right_cell.find_all("li")
        if lis:
            bullets = [li.get_text(" ", strip=True).strip() for li in lis if li.get_text(strip=True)]
            right_text = " ".join(bullets).strip()
        else:
            right_text = _cell_text(right_cell).strip()

        if not right_text:
            continue

        text_core = f"{header2} {right_text}".strip() if header2 else right_text
        text = f"{title} {text_core}".strip()
        out.append({"title": title, "text": text})

    return out


def extract_sections_from_main(soup: BeautifulSoup) -> list[dict]:
    """
    Extract sections from <main>, bounded strictly by h2 headings.
    Fixes duplicate/leakage by:
      - stopping at the NEXT h2 encountered (any h2)
      - collecting only leaf text blocks (p, li), not wrapper div/ul/ol/span
    """
    main = soup.find("main") or soup.find("article") or soup.body
    if not main:
        return []

    # remove <section class="pagedetails"> within main
    for pd in main.find_all("section", class_=lambda c: c and "pagedetails" in str(c).split()):
        pd.decompose()

    h2s = [h for h in main.find_all("h2") if isinstance(h, Tag)]
    sections: list[dict] = []

    for h2 in h2s:
        if not _is_within_main(h2, main):
            continue

        h2_title = h2.get_text(" ", strip=True)
        if not h2_title:
            continue

        # skip TOC blocks if present
        if h2_title.strip().lower() == "table of contents":
            continue

        is_special = _h2_is_authorities_or_responsibilities(h2_title)

        # Find first table between this h2 and the next h2 (if any)
        first_table: Tag | None = None
        for el in h2.next_elements:
            if isinstance(el, Tag) and el.name == "h2":
                break
            if isinstance(el, Tag) and el.name == "table" and _is_within_main(el, main):
                first_table = el
                break

        if is_special and first_table is not None:
            sections.extend(_parse_table_authorities_responsibilities(first_table))
            continue

        parts: list[str] = []
        processed_table_ids: set[int] = set()

        for el in h2.next_elements:
            if not isinstance(el, Tag):
                continue
            if not _is_within_main(el, main):
                continue

            # STOP at the next section header (any h2)
            if el.name == "h2":
                break

            # Never collect raw text from inside tables.
            parent_table = el.find_parent("table")
            if parent_table is not None and el.name != "table":
                continue

            if el.name == "table":
                if id(el) in processed_table_ids:
                    continue
                processed_table_ids.add(id(el))
                parts.extend(_format_table_inline(el))
                continue

            # ✅ Only leaf-level text blocks
            if el.name in {"p", "li"}:
                if el.find("table") is not None:
                    continue
                txt = el.get_text(" ", strip=True)
                if txt:
                    parts.append(txt)

        # de-dup exact repeats
        seen: set[str] = set()
        dedup_parts: list[str] = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                dedup_parts.append(p)

        body_text = re.sub(r"\s+", " ", " ".join(dedup_parts)).strip()
        if body_text:
            sections.append({"title": h2_title, "text": body_text})

    return sections


def build_document_record(page_url: str, html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")

    canonical_url = extract_canonical_url(soup, page_url)

    title = extract_title(soup)
    if not title:
        # fallback to <title> or <h1>
        if soup.title and soup.title.get_text(strip=True):
            title = soup.title.get_text(" ", strip=True)
        else:
            h1 = soup.find("h1")
            title = h1.get_text(" ", strip=True) if h1 else None

    author = extract_author(soup)
    effective_date, published_date = extract_dates(soup)
    language = extract_language(soup)
    subjects = extract_subjects(soup)

    sections = extract_sections_from_main(soup)

    doc_text = " ".join(
        f"{s.get('title','')} {s.get('text','')}".strip()
        for s in sections
        if (s.get("title") or s.get("text"))
    ).strip()

    doc_hash = sha256_text(doc_text)

    doc = {
        "id": stable_id_from_url(canonical_url),
        "url": page_url,
        "domain": urlparse(page_url).netloc,
        "title": title,
        "author": author,
        "effective_date": effective_date,
        "published_date": published_date,
        "retrieved_at": now_iso(),
        "subjects": subjects,
        "language": language,
        "canonical_url": canonical_url,
        "hash": doc_hash,
        "sections": sections,
        "raw_html": None,
        "text": doc_text,
    }
    return doc


def append_jsonl(path: str, obj: dict) -> None:
    # kept (unused) to minimize changes
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    items = load_catalog_items(CATALOG_PATH)

    # If state is Cancelled, force status=skipped and never parse it
    for it in items:
        if (it.get("state") or "").strip().lower() == "cancelled":
            update_status(CATALOG_PATH, it["url"], "skipped")
            it["status"] = "skipped"

    pending = [it for it in items if it["status"] == "pending"]

    print(f"Catalog items: {len(items)} | pending: {len(pending)}")

    with requests.Session() as session:
        for it in pending:
            page_url = it["url"]
            try:
                html = fetch_html(page_url, session)
                doc = build_document_record(page_url, html)

                out_path = write_pretty_json(OUT_JSON_DIR, doc)

                update_status(CATALOG_PATH, page_url, "parsed")

                print(f"parsed: {page_url}")
                print(f"saved:  {out_path}")
            except Exception as e:
                update_status(CATALOG_PATH, page_url, "skipped")
                print(f"skipped: {page_url} ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
