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

FULLTEXT_INDEX = os.getenv("FULLTEXT_INDEX", "fulltextcontent")
CANDIDATE_POOL = int(os.getenv("KEYWORD_CANDIDATE_POOL", "80"))

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


# ---- Keywords ----
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


def _extract_keywords(question: str, max_keywords: int = 10) -> List[str]:
    text = _normalize(question)
    tokens = text.split()

    out: List[str] = []
    seen = set()

    for t in tokens:
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


def _build_fulltext_query(keywords: List[str]) -> str:
    return " OR ".join([f"\"{kw}\"" if " " in kw else kw for kw in keywords])


def _contains_kw(text: str, kw: str) -> bool:
    return re.search(rf"\b{re.escape(kw)}\b", text, flags=re.IGNORECASE) is not None


def _hit_count(blob: str, keywords: List[str]) -> int:
    return sum(1 for kw in keywords if _contains_kw(blob, kw))


def retrieve_keyword(question: str, top_k: int = 8) -> Dict[str, Any]:
    q = (question or "").strip()
    if not q:
        return {"retriever": "keyword_hits", "question": question, "top_k": top_k, "results": []}

    ref = parse_daod_ref(q)
    daod_base = ref["daod_base"]
    daod_full = ref["daod_full"]

    keywords = _extract_keywords(q, max_keywords=10)
    if not keywords and not daod_base and not daod_full:
        return {"retriever": "keyword_hits", "question": question, "top_k": top_k, "results": []}

    ft_query = _build_fulltext_query(keywords) if keywords else (daod_full or daod_base)

    cypher = """
    CALL db.index.fulltext.queryNodes($index_name, $q) YIELD node, score
    WITH node, score
    WHERE node:Chunk

    OPTIONAL MATCH (s:Section)-[:CONTAINS]->(node)
    OPTIONAL MATCH (d:Document)-[:CONTAINS]->(s)
    OPTIONAL MATCH (d2:Document)-[:CONTAINS]->(node)

    WITH node, score,
         coalesce(d.title, d2.title, "") AS doc_title,
         coalesce(d.daod_number, d2.daod_number, "") AS daod_number,
         coalesce(s.title, node.section, "") AS section_title,
         coalesce(node.text, "") AS text

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

    RETURN
      node.id AS chunk_id,
      text AS text,
      score AS ft_score,
      doc_title AS doc_title,
      daod_number AS daod_number,
      section_title AS section_title
    ORDER BY ft_score DESC
    LIMIT $candidate_pool
    """

    with driver.session() as session:
        rows = session.run(
            cypher,
            index_name=FULLTEXT_INDEX,
            q=ft_query,
            daod_base=daod_base,
            daod_full=daod_full,
            candidate_pool=int(CANDIDATE_POOL),
        ).data()

    scored: List[Dict[str, Any]] = []
    for r in rows:
        blob = " ".join([r.get("doc_title", ""), r.get("daod_number", ""), r.get("section_title", ""), r.get("text", "")])
        hc = _hit_count(blob, keywords) if keywords else 0
        scored.append(
            {
                "chunk_id": r.get("chunk_id", ""),
                "text": r.get("text", ""),
                "score": float(r.get("ft_score", 0.0)),
                "hit_count": int(hc),
                "doc_title": r.get("doc_title", ""),
                "daod_number": r.get("daod_number", ""),
                "section_title": r.get("section_title", ""),
            }
        )

    scored.sort(key=lambda x: (x["hit_count"], x["score"]), reverse=True)

    return {
        "retriever": "keyword_hits",
        "question": question,
        "top_k": int(top_k),
        "daod_base": daod_base,
        "daod_full": daod_full,
        "keywords": keywords,
        "results": scored[: int(top_k)],
    }


if __name__ == "__main__":
    q = input("Question> ").strip()
    print(retrieve_keyword(q, top_k=8))
