from lxml import etree

CATALOG_PATH = "./data/catalog.xml"


def reset_all_to_pending(xml_path: str) -> int:
    """
    Sets <status>pending</status> for EVERY <Item> in the catalog.
    Creates <status> if missing.
    Returns the number of <Item> elements processed.
    """
    tree = etree.parse(xml_path)
    root = tree.getroot()

    count = 0
    for item in root.findall("Item"):
        status_el = item.find("status")
        if status_el is None:
            status_el = etree.SubElement(item, "status")
        status_el.text = "pending"
        count += 1

    tree.write(xml_path, pretty_print=True, xml_declaration=True, encoding="utf-8")
    return count


def main() -> None:
    n = reset_all_to_pending(CATALOG_PATH)
    print(f"Reset complete: {n} item(s) set to pending in {CATALOG_PATH}")


if __name__ == "__main__":
    main()