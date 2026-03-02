import os
from typing import Any, Dict, List

from neo4j import GraphDatabase

###
from pathlib import Path
from dotenv import load_dotenv
 
# Find project root dynamically
BASE_DIR = Path(__file__).resolve().parents[2]
 
# Load .env from root
load_dotenv(BASE_DIR / ".env")
###

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

SIBLING_WINDOW = int(os.getenv("SIBLING_WINDOW", "2"))

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def expand_to_section(chunks: List[Dict[str, Any]], window: int = SIBLING_WINDOW) -> List[Dict[str, Any]]:
    """
    Expand seed chunks to include nearby sibling chunks in the same Section (± window).
    Graph schema:
      (d:Document)-[:HAS_SECTION]-(s:Section)-[:CONTAINS]-(c:Chunk)
    """
    seed_ids = [c.get("chunk_id") for c in chunks if c.get("chunk_id")]
    if not seed_ids:
        return []

    cypher = """
    UNWIND $seed_ids AS cid
    MATCH (hit:Chunk {id: cid})

    OPTIONAL MATCH (s:Section)-[:CONTAINS]-(hit)
    OPTIONAL MATCH (d:Document)-[:HAS_SECTION]-(s)

    WITH hit, s, d,
         coalesce(hit.chunk_index, hit.chunkindex, null) AS idx,
         coalesce(hit.start_char, hit.startchar, null) AS startc

    CALL {
      WITH hit, s, idx, startc
      WITH hit, s, idx, startc, $window AS window

      OPTIONAL MATCH (s)-[:CONTAINS]-(sib:Chunk)
      WITH hit, s, idx, startc, window, collect(DISTINCT sib) AS sibs

      WITH hit, s, idx, startc, window,
           [x IN sibs
            WHERE s IS NOT NULL AND (
              (idx IS NOT NULL AND
                 coalesce(x.chunk_index, x.chunkindex, -999999) >= idx - window AND
                 coalesce(x.chunk_index, x.chunkindex,  999999) <= idx + window
              )
              OR
              (idx IS NULL AND startc IS NOT NULL AND
                 coalesce(x.start_char, x.startchar, -999999999) >= startc - 2000 * window AND
                 coalesce(x.start_char, x.startchar,  999999999) <= startc + 2000 * window
              )
            )
           ] AS filtered

      WITH CASE
             WHEN s IS NULL OR size(filtered) = 0 THEN [hit]
             ELSE filtered
           END AS chosen

      UNWIND chosen AS c
      RETURN DISTINCT c AS c
    }

    // Force ONE row per chunk (prevents blank doc_title winning)
    OPTIONAL MATCH (s2:Section)-[:CONTAINS]-(c)
    OPTIONAL MATCH (d2:Document)-[:HAS_SECTION]-(s2)

    WITH c,
         head(collect(DISTINCT d2.title)) AS doc_title,
         head(collect(DISTINCT d2.daod_number)) AS daod_number,
         head(collect(DISTINCT s2.title)) AS section_title

    RETURN
      c.id AS chunk_id,
      coalesce(c.text, "") AS text,
      coalesce(doc_title, "") AS doc_title,
      coalesce(daod_number, "") AS daod_number,
      coalesce(section_title, c.section, "") AS section_title,
      coalesce(c.chunk_index, c.chunkindex, null) AS chunk_index,
      coalesce(c.start_char, c.startchar, null) AS start_char
    """

    with driver.session() as session:
        rows = session.run(cypher, seed_ids=seed_ids, window=int(window)).data()

    # safe dedupe by chunk_id
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        cid = r.get("chunk_id")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append(r)

    return out
