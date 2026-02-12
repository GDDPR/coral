from lxml import etree

# Updates <status> for the <Item> whose <url> exactly equals exact_url. Returns True if updated, False if URL not found.
def update_status(xml_path: str, exact_url: str, new_status: str) -> bool:
    tree = etree.parse(xml_path)
    root = tree.getroot()

    for item in root.findall("Item"):
        url_el = item.find("url")
        if url_el is None:
            continue
        if (url_el.text or "").strip() == exact_url.strip():
            status_el = item.find("status")
            if status_el is None:
                status_el = etree.SubElement(item, "status")
            status_el.text = new_status
            tree.write(xml_path, pretty_print=True, xml_declaration=True, encoding="utf-8")
            return True

    return False
