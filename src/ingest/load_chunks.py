import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from neo4j import GraphDatabase


DOCS_DIR_DEFAULT = Path("data/docs_json")
CHUNKS_JSONL_DEFAULT = Path("data/chunks.jsonl")
SECTIONS_JSONL_DEFAULT = Path("data/sections.jsonl")
LINKS_JSONL_DEFAULT = Path("data/chunk_links.jsonl")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def batched(iterable: Iterable[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> None:
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")

    if not password:
        raise RuntimeError(
            "Missing NEO4J_PASSWORD env var.\n"
            "Example (PowerShell): $env:NEO4J_PASSWORD='yourpass'\n"
            "Example (bash): export NEO4J_PASSWORD='yourpass'"
        )

    docs_dir = Path(os.environ.get("DOCS_DIR", str(DOCS_DIR_DEFAULT)))
    chunks_path = Path(os.environ.get("CHUNKS_JSONL", str(CHUNKS_JSONL_DEFAULT)))
    sections_path = Path(os.environ.get("SECTIONS_JSONL", str(SECTIONS_JSONL_DEFAULT)))
    links_path = Path(os.environ.get("CHUNK_LINKS_JSONL", str(LINKS_JSONL_DEFAULT)))
    batch_size = int(os.environ.get("BATCH_SIZE", "300"))

    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir.resolve()}")

    for p, label in [
        (chunks_path, "Chunks JSONL"),
        (sections_path, "Sections JSONL"),
        (links_path, "Chunk links JSONL"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p.resolve()}")

    doc_files = sorted(docs_dir.glob("*.json"))
    if not doc_files:
        raise FileNotFoundError(f"No .json files found in: {docs_dir.resolve()}")

    print(f"Neo4j URI: {uri}")
    print(f"Docs: {len(doc_files)} files in {docs_dir}")
    print(f"Chunks:   {chunks_path}")
    print(f"Sections: {sections_path}")
    print(f"Links:    {links_path}")
    print(f"Batch size: {batch_size}")

    driver = GraphDatabase.driver(uri, auth=(user, password))

    # Constraints (prof-aligned + Section.id)
    create_constraints = [
        """
        CREATE CONSTRAINT docid IF NOT EXISTS
        FOR (d:Document) REQUIRE d.id IS UNIQUE
        """,
        """
        CREATE CONSTRAINT sectionid IF NOT EXISTS
        FOR (s:Section) REQUIRE s.id IS UNIQUE
        """,
        """
        CREATE CONSTRAINT chunkid IF NOT EXISTS
        FOR (c:Chunk) REQUIRE c.id IS UNIQUE
        """,
    ]

    doc_upsert = """
    MERGE (d:Document {id: $id})
    SET d.url = $url,
        d.canonical_url = $canonical_url,
        d.domain = $domain,
        d.title = $title,
        d.author = $author,
        d.effective_date = $effective_date,
        d.published_date = $published_date,
        d.retrieved_at = $retrieved_at,
        d.language = $language,
        d.hash = $hash,
        d.subjects = $subjects,
        d.raw_html = $raw_html,
        d.text = $text
    """

    # 1) Load Chunk nodes from chunks.jsonl (prof schema only)
    chunk_nodes_upsert = """
    UNWIND $rows AS row
    MERGE (c:Chunk {id: row.id})
    SET c.text = row.text,
        c.start_char = row.start_char,
        c.end_char = row.end_char,
        c.section = row.section,
        c.token_count = row.token_count,
        c.embedding = row.embedding
    """

    # 2) Load Section nodes and connect Document -> Section
    sections_upsert = """
    UNWIND $rows AS row
    MATCH (d:Document {id: row.doc_id})
    MERGE (s:Section {id: row.id})
    SET s.title = row.title,
        s.section_index = row.section_index
    MERGE (d)-[:HAS_SECTION]->(s)
    """

    # 3) Connect Section -> Chunk using chunk_links.jsonl
    links_upsert = """
    UNWIND $rows AS row
    MATCH (s:Section {id: row.section_id})
    MATCH (c:Chunk {id: row.chunk_id})
    MERGE (s)-[:CONTAINS]->(c)
    """

    with driver:
        for q in create_constraints:
            driver.execute_query(q)
        print("Constraints ensured.")

        # Documents
        docs_loaded = 0
        for p in doc_files:
            doc = read_json(p)

            doc_id = doc.get("id")
            if not doc_id:
                print(f"[SKIP] {p.name}: missing 'id'")
                continue

            driver.execute_query(
                doc_upsert,
                id=doc_id,
                url=doc.get("url"),
                canonical_url=doc.get("canonical_url"),
                domain=doc.get("domain"),
                title=doc.get("title"),
                author=doc.get("author"),
                effective_date=doc.get("effective_date"),
                published_date=doc.get("published_date"),
                retrieved_at=doc.get("retrieved_at"),
                language=doc.get("language"),
                hash=doc.get("hash"),
                subjects=doc.get("subjects", []),
                raw_html=doc.get("raw_html"),
                text=doc.get("text"),
            )
            docs_loaded += 1
        print(f"Documents upserted: {docs_loaded}")

        # Chunk nodes
        chunk_nodes_loaded = 0
        for batch in batched(iter_jsonl(chunks_path), batch_size=batch_size):
            driver.execute_query(chunk_nodes_upsert, rows=batch)
            chunk_nodes_loaded += len(batch)
        print(f"Chunk nodes upserted: {chunk_nodes_loaded}")

        # Sections
        sections_loaded = 0
        for batch in batched(iter_jsonl(sections_path), batch_size=batch_size):
            driver.execute_query(sections_upsert, rows=batch)
            sections_loaded += len(batch)
        print(f"Sections upserted: {sections_loaded}")

        # Links
        links_loaded = 0
        for batch in batched(iter_jsonl(links_path), batch_size=batch_size):
            driver.execute_query(links_upsert, rows=batch)
            links_loaded += len(batch)
        print(f"Section->Chunk links upserted: {links_loaded}")

    print("Done. Graph is loaded: Document -> Section -> Chunk.")


if __name__ == "__main__":
    main()