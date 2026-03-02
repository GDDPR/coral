import os
import re
from typing import Any, Dict, List, Set

from neo4j import GraphDatabase

###
from pathlib import Path
from dotenv import load_dotenv
 
# Find project root dynamically
BASE_DIR = Path(__file__).resolve().parents[2]
 
# Load .env from root
load_dotenv(BASE_DIR / ".env")
###

# ---- Neo4j ----
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ---- DAOD parsing ----
DAOD_SPEC_PATTERN = re.compile(r"\bDAOD\s*(6\d{3})\s*[-–—]?\s*(\d)\b", re.IGNORECASE)
DAOD_BASE_PATTERN = re.compile(r"\bDAOD\s*(6\d{3})\b", re.IGNORECASE)
DAOD_FIVE_DIGIT_PATTERN = re.compile(r"\bDAOD\s*(6\d{4})\b", re.IGNORECASE)

def parse_daod_ref(text: str) -> Dict[str, str | None]:
    t = text or ""

    m = DAOD_SPEC_PATTERN.search(t)
    if m:
        base, part = m.group(1), m.group(2)
        return {"daod_base": base, "daod_full": f"{base}-{part}"}

    m2 = DAOD_FIVE_DIGIT_PATTERN.search(t)
    if m2:
        five = m2.group(1)
        base, part = five[:4], five[4:]
        if part.isdigit() and len(part) == 1:
            return {"daod_base": base, "daod_full": f"{base}-{part}"}

    m3 = DAOD_BASE_PATTERN.search(t)
    if m3:
        return {"daod_base": m3.group(1), "daod_full": None}

    return {"daod_base": None, "daod_full": None}


# ---- Keyword extraction for entity matching ----
STOPWORDS: Set[str] = {
    "the","a","an","and","or","but","to","of","in","on","for","with",
    "is","are","was","were","be","been","being","as","at","by","it",
    "this","that","these","those","from","into","about","over","under",
    "i","you","we","they","he","she","them","us","my","your","our",
}

def _normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _extract_keywords(question: str, max_keywords: int = 12) -> List[str]:
    text = _normalize(question)
    toks = text.split()

    out: List[str] = []
    seen = set()
    for t in toks:
        if t in STOPWORDS:
            continue
        if len(t) < 3:
            continue
        if t.isdigit():
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= max_keywords:
            break
    return out


def retrieve_entity(question: str, top_k: int = 8) -> Dict[str, Any]:
    q = (question or "").strip()
    if not q:
        return {"retriever": "entity_graph", "question": question, "top_k": top_k, "results": []}

    ref = parse_daod_ref(q)
    daod_base = ref["daod_base"]
    daod_full = ref["daod_full"]

    keywords = _extract_keywords(q, max_keywords=12)
    if not keywords and not daod_base and not daod_full:
        return {"retriever": "entity_graph", "question": question, "top_k": top_k, "results": []}

    cypher = """
    WITH $keywords AS kws

    MATCH (e:Entity)
    WITH e, kws,
         toLower(coalesce(e.name, "")) AS ename,
         [a IN coalesce(e.aliases, []) | toLower(a)] AS ealiases

    WITH e, kws,
         reduce(s = 0, kw IN kws |
           s + CASE
                 WHEN ename CONTAINS kw OR any(al IN ealiases WHERE al CONTAINS kw) THEN 1
                 ELSE 0
               END
         ) AS entity_score
    WHERE entity_score > 0
    ORDER BY entity_score DESC
    LIMIT $max_entities

    MATCH (c:Chunk)-[:MENTIONS]->(e)
    WITH c, sum(entity_score) AS chunk_entity_score, collect(e.name)[0..10] AS matched_entities

    OPTIONAL MATCH (s:Section)-[:CONTAINS]->(c)
    OPTIONAL MATCH (d:Document)-[:CONTAINS]->(s)
    OPTIONAL MATCH (d2:Document)-[:CONTAINS]->(c)

    WITH c, chunk_entity_score, matched_entities,
         coalesce(d.title, d2.title, "") AS doc_title,
         coalesce(d.daod_number, d2.daod_number, "") AS daod_number,
         coalesce(s.title, c.section, "") AS section_title,
         coalesce(c.text, "") AS chunk_text

    WHERE
    (
      $daod_full IS NULL AND $daod_base IS NULL
    )
    OR
    (
      $daod_full IS NOT NULL AND (daod_number = $daod_full OR doc_title CONTAINS $daod_full)
    )
    OR
    (
      $daod_full IS NULL AND $daod_base IS NOT NULL AND
      (
        daod_number STARTS WITH ($daod_base + "-")
        OR doc_title CONTAINS ($daod_base + "-")
        OR doc_title CONTAINS $daod_base
      )
    )

    WITH c, chunk_entity_score, matched_entities, doc_title, daod_number, section_title, chunk_text,
         reduce(s = 0, kw IN $keywords |
           s + CASE WHEN toLower(chunk_text) CONTAINS kw THEN 1 ELSE 0 END
         ) AS hit_bonus

    WITH c, matched_entities, doc_title, daod_number, section_title, chunk_text,
         (chunk_entity_score * 10 + hit_bonus) AS final_score,
         chunk_entity_score AS entity_score,
         hit_bonus AS hit_bonus

    RETURN
      c.id AS chunk_id,
      chunk_text AS text,
      final_score AS score,
      entity_score AS entity_score,
      hit_bonus AS hit_bonus,
      matched_entities AS matched_entities,
      doc_title AS doc_title,
      daod_number AS daod_number,
      section_title AS section_title
    ORDER BY score DESC
    LIMIT $limit
    """

    with driver.session() as session:
        rows = session.run(
            cypher,
            keywords=keywords,
            daod_base=daod_base,
            daod_full=daod_full,
            max_entities=20,
            limit=int(top_k),
        ).data()

    results = []
    for r in rows:
        results.append(
            {
                "chunk_id": r.get("chunk_id", ""),
                "text": r.get("text", ""),
                "score": float(r.get("score", 0.0)),
                "doc_title": r.get("doc_title", ""),
                "daod_number": r.get("daod_number", ""),
                "section_title": r.get("section_title", ""),
                "entity_score": float(r.get("entity_score", 0.0)),
                "hit_bonus": int(r.get("hit_bonus", 0)),
                "matched_entities": r.get("matched_entities", []),
            }
        )

    return {
        "retriever": "entity_graph",
        "question": question,
        "top_k": int(top_k),
        "daod_base": daod_base,
        "daod_full": daod_full,
        "keywords": keywords,
        "results": results,
    }


if __name__ == "__main__":
    q = input("Question> ").strip()
    print(retrieve_entity(q, top_k=8))
