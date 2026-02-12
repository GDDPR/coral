import hashlib
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

from daod6000_scraper import scrape_daod6000_items


def now_z() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def mkid(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    """
    Pretty-print helper for ElementTree (so XML isn't one giant line).
    Works on all Python versions.
    """
    i = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent_xml(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def addtocatalog(items: list[dict], catalogpath: str = "catalog.xml") -> None:
    try:
        tree = ET.parse(catalogpath)
        root = tree.getroot()
    except FileNotFoundError:
        root = ET.Element("Catalog", {"createdat": now_z()})
        tree = ET.ElementTree(root)

    existing = {e.findtext("url") for e in root.findall("Item")}

    for it in items:
        url = it["url"]
        if url in existing:
            continue

        item_el = ET.SubElement(root, "Item")
        ET.SubElement(item_el, "id").text = mkid(url)
        ET.SubElement(item_el, "url").text = url
        ET.SubElement(item_el, "domain").text = urlparse(url).netloc
        ET.SubElement(item_el, "discoveredat").text = now_z()
        ET.SubElement(item_el, "status").text = "pending"

        state = it.get("state") or ""
        ET.SubElement(item_el, "state").text = state

    # ✅ Make it pretty before writing
    indent_xml(root)

    tree.write(catalogpath, encoding="utf-8", xml_declaration=True)


def main() -> None:
    items = scrape_daod6000_items()
    addtocatalog(items, catalogpath="./data/catalog.xml")
    print(f"Merged {len(items)} scraped items into ./data/catalog.xml")


if __name__ == "__main__":
    main()
