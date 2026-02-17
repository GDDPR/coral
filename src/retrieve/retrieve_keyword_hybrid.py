import os
import re
from typing import Any, Dict, List, Set

import requests
from neo4j import GraphDatabase

# -------------------------
# Config (env overrides)
# -------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

VECTOR_INDEX = os.getenv("VECTOR_INDEX", "chunkembedding")
FULLTEXT_INDEX = os.getenv("FULLTEXT_INDEX", "fulltextcontent")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "60"))
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "768"))

VEC_CANDIDATES = int(os.getenv("HYBRID_VEC_CANDIDATES", "25"))
FT_CANDIDATES = int(os.getenv("HYBRID_FT_CANDIDATES", "60"))

RRF_K = int(os.getenv("HYBRID_RRF_K", "60"))
W_VEC = float(os.getenv("HYBRID_W_VEC", "1.0"))
W_FT = float(os.getenv("HYBRID_W_FT", "0.8"))
W_KW_BONUS = float(os.getenv("HYBRID_W_KW_BONUS", "0.15"))

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# -------------------------
# DAOD parsing (6xxx-x + typo 60001)
# -------------------------
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
        five = m2.group(1)  # e.g. 60001
        base, part = five[:4], five[4:]
        if part.isdigit() and len(part) == 1:
            return {"daod_base": base, "daod_full": f"{base}-{part}"}

    m3 = DAOD_BASE_PATTERN.search(t)
    if m3:
        return {"daod_base": m3.group(1), "daod_full": None}

    return {"daod_base": None, "daod_full": None}


# -------------------------
# Keyword helpers
# -------------------------
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


def _build_fulltext_query(keywords: List[str]) -> str:
    return " OR ".join([f"\"{kw}\"" if " " in kw else kw for kw in keywords])


def _contains_kw(text: str, kw: str) -> bool:
    return re.search(rf"\b{re.escape(kw)}\b", text, flags=re.IGNORECASE) is not None


def _hit_count(blob: str, keywords: List[str]) -> int:
    return sum(1 for kw in keywords if _contains_kw(blob, kw))


# -------------------------
# Ollama embeddings
# -------------------------
def _ollama_embed(text: str) -> List[float]:
    payload = {"model": OLLAMA_EMBED_MODEL, "prompt": text}

    try:
        r = requests.post(f"{OLLAMA_HOST}/api/embeddings", json=payload, timeout=HTTP_TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            emb = data.get("embedding")
            if isinstance(emb, list) and emb:
                return emb
    except requests.RequestException:
        pass

    try:
        r = requests.post(f"{OLLAMA_HOST}/api/embed", json=payload, timeout=HTTP_TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            if "embedding" in data and isinstance(data["embedding"], list):
                return data["embedding"]
            if "embeddings" in data and isinstance(data["embeddings"], list) and data["embeddings"]:
                return data["embeddings"][0]
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama embed request failed: {e}")

    raise RuntimeError("Could not get embedding from Ollama (tried /api/embeddings and /api/embed).")


# -------------------------
# Neo4j retrieval
# -------------------------
def _vector_candidates(question: str, limit: int) -> List[Dict[str, Any]]:
    emb = _ollama_embed(question)

    cypher = """
    CALL db.index.vector.queryNodes($index_name, $k, $embedding) YIELD node, score
    WITH node, score
    WHERE node:Chunk

    OPTIONAL MATCH (s:Section)-[:CONTAINS]-(node)
    OPTIONAL MATCH (d:Document)-[:HAS_SECTION]-(s)

    WITH node, score,
         head(collect(DISTINCT d.title)) AS doc_title,
         head(collect(DISTINCT d.daod_number)) AS daod_number,
         head(collect(DISTINCT s.title)) AS section_title,
         coalesce(node.text, "") AS text,
         coalesce(node.chunk_index, node.chunkindex, null) AS chunk_index,
         coalesce(node.start_char, node.startchar, null) AS start_char

    RETURN
      node.id AS chunk_id,
      text AS text,
      score AS vec_score,
      coalesce(doc_title, "") AS doc_title,
      coalesce(daod_number, "") AS daod_number,
      coalesce(section_title, node.section, "") AS section_title,
      chunk_index AS chunk_index,
      start_char AS start_char
    ORDER BY vec_score DESC
    LIMIT $limit
    """

    with driver.session() as session:
        rows = session.run(
            cypher,
            index_name=VECTOR_INDEX,
            k=int(limit),
            embedding=emb,
            limit=int(limit),
        ).data()

    return rows


def _fulltext_candidates(keywords: List[str], daod_base: str | None, daod_full: str | None, limit: int) -> List[Dict[str, Any]]:
    if not keywords and not daod_base and not daod_full:
        return []

    q = _build_fulltext_query(keywords) if keywords else (daod_full or daod_base)

    cypher = """
    CALL db.index.fulltext.queryNodes($index_name, $q) YIELD node, score
    WITH node, score
    WHERE node:Chunk

    OPTIONAL MATCH (s:Section)-[:CONTAINS]-(node)
    OPTIONAL MATCH (d:Document)-[:HAS_SECTION]-(s)

    WITH node, score,
         head(collect(DISTINCT d.title)) AS doc_title,
         head(collect(DISTINCT d.daod_number)) AS daod_number,
         head(collect(DISTINCT s.title)) AS section_title,
         coalesce(node.text, "") AS text,
         coalesce(node.chunk_index, node.chunkindex, null) AS chunk_index,
         coalesce(node.start_char, node.startchar, null) AS start_char

    WHERE
    (
      $daod_full IS NULL AND $daod_base IS NULL
    )
    OR
    (
      $daod_full IS NOT NULL AND (coalesce(daod_number, "") = $daod_full OR coalesce(doc_title, "") CONTAINS $daod_full)
    )
    OR
    (
      $daod_full IS NULL AND $daod_base IS NOT NULL AND
      (
        coalesce(doc_title, "") CONTAINS ($daod_base + "-")
        OR coalesce(doc_title, "") CONTAINS $daod_base
        OR coalesce(daod_number, "") STARTS WITH ($daod_base + "-")
      )
    )

    RETURN
      node.id AS chunk_id,
      text AS text,
      score AS ft_score,
      coalesce(doc_title, "") AS doc_title,
      coalesce(daod_number, "") AS daod_number,
      coalesce(section_title, node.section, "") AS section_title,
      chunk_index AS chunk_index,
      start_char AS start_char
    ORDER BY ft_score DESC
    LIMIT $limit
    """

    with driver.session() as session:
        rows = session.run(
            cypher,
            index_name=FULLTEXT_INDEX,
            q=q,
            daod_base=daod_base,
            daod_full=daod_full,
            limit=int(limit),
        ).data()

    return rows


# -------------------------
# Merge + rerank
# -------------------------
def _add_rrf(scores: Dict[str, float], ranked_chunk_ids: List[str], weight: float, rrf_k: int) -> None:
    for i, cid in enumerate(ranked_chunk_ids, start=1):
        scores[cid] = scores.get(cid, 0.0) + weight * (1.0 / (rrf_k + i))


def retrieve_keyword_hybrid(question: str, top_k: int = 8) -> Dict[str, Any]:
    q = (question or "").strip()
    if not q:
        return {"retriever": "keyword_hybrid", "question": question, "top_k": top_k, "results": []}

    ref = parse_daod_ref(q)
    daod_base = ref["daod_base"]
    daod_full = ref["daod_full"]

    keywords = _extract_keywords(q, max_keywords=10)

    vec_rows = _vector_candidates(q, limit=VEC_CANDIDATES)
    ft_rows = _fulltext_candidates(keywords, daod_base=daod_base, daod_full=daod_full, limit=FT_CANDIDATES)

    merged: Dict[str, Dict[str, Any]] = {}

    for r in vec_rows:
        cid = r.get("chunk_id", "")
        if not cid:
            continue
        merged[cid] = {
            "chunk_id": cid,
            "text": r.get("text", ""),
            "doc_title": r.get("doc_title", ""),
            "daod_number": r.get("daod_number", ""),
            "section_title": r.get("section_title", ""),
            "chunk_index": r.get("chunk_index", None),
            "start_char": r.get("start_char", None),
            "vec_score": float(r.get("vec_score", 0.0)),
            "ft_score": 0.0,
        }

    for r in ft_rows:
        cid = r.get("chunk_id", "")
        if not cid:
            continue
        if cid not in merged:
            merged[cid] = {
                "chunk_id": cid,
                "text": r.get("text", ""),
                "doc_title": r.get("doc_title", ""),
                "daod_number": r.get("daod_number", ""),
                "section_title": r.get("section_title", ""),
                "chunk_index": r.get("chunk_index", None),
                "start_char": r.get("start_char", None),
                "vec_score": 0.0,
                "ft_score": float(r.get("ft_score", 0.0)),
            }
        else:
            merged[cid]["ft_score"] = max(merged[cid].get("ft_score", 0.0), float(r.get("ft_score", 0.0)))
            for k in ["text", "doc_title", "daod_number", "section_title", "chunk_index", "start_char"]:
                if merged[cid].get(k) in (None, "") and r.get(k) not in (None, ""):
                    merged[cid][k] = r.get(k)

    fused_scores: Dict[str, float] = {}
    vec_ranked = [r["chunk_id"] for r in vec_rows if r.get("chunk_id")]
    ft_ranked = [r["chunk_id"] for r in ft_rows if r.get("chunk_id")]

    _add_rrf(fused_scores, vec_ranked, weight=W_VEC, rrf_k=RRF_K)
    _add_rrf(fused_scores, ft_ranked, weight=W_FT, rrf_k=RRF_K)

    for cid, item in merged.items():
        blob = " ".join(
            [
                item.get("doc_title", ""),
                item.get("daod_number", ""),
                item.get("section_title", ""),
                item.get("text", ""),
            ]
        )
        hc = _hit_count(blob, keywords) if keywords else 0
        item["hit_count"] = int(hc)
        item["score"] = float(fused_scores.get(cid, 0.0) + W_KW_BONUS * hc)

    ranked = sorted(merged.values(), key=lambda x: x.get("score", 0.0), reverse=True)[: int(top_k)]

    return {
        "retriever": "keyword_hybrid",
        "question": question,
        "top_k": int(top_k),
        "daod_base": daod_base,
        "daod_full": daod_full,
        "keywords": keywords,
        "results": [
            {
                "chunk_id": r["chunk_id"],
                "text": r.get("text", ""),
                "score": float(r.get("score", 0.0)),
                "doc_title": r.get("doc_title", ""),
                "daod_number": r.get("daod_number", ""),
                "section_title": r.get("section_title", ""),
                "chunk_index": r.get("chunk_index", None),
                "start_char": r.get("start_char", None),
                # debug
                "hit_count": int(r.get("hit_count", 0)),
                "vec_score": float(r.get("vec_score", 0.0)),
                "ft_score": float(r.get("ft_score", 0.0)),
            }
            for r in ranked
        ],
    }


if __name__ == "__main__":
    q = input("Question> ").strip()
    data = retrieve_keyword_hybrid(q, top_k=8)
    print(data)
