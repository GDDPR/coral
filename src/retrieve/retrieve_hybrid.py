import os
from typing import Any, Dict, List

import requests
from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

VECTOR_INDEX = os.getenv("VECTOR_INDEX", "chunkembedding")
FULLTEXT_INDEX = os.getenv("FULLTEXT_INDEX", "fulltextcontent")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "60"))

VEC_CANDIDATES = int(os.getenv("HYBRID_VEC_CANDIDATES", "25"))
FT_CANDIDATES = int(os.getenv("HYBRID_FT_CANDIDATES", "60"))

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def _ollama_embed(text: str) -> List[float]:
    payload = {"model": OLLAMA_EMBED_MODEL, "prompt": text}

    r = requests.post(f"{OLLAMA_HOST}/api/embeddings", json=payload, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    emb = data.get("embedding")
    if not isinstance(emb, list) or not emb:
        raise RuntimeError("Ollama did not return an embedding list")
    return emb


def _vector_candidates(question: str, limit: int) -> List[Dict[str, Any]]:
    emb = _ollama_embed(question)

    cypher = """
    CALL db.index.vector.queryNodes($index_name, $k, $embedding) YIELD node, score
    WITH node, score
    WHERE node:Chunk

    OPTIONAL MATCH (s:Section)-[:CONTAINS]->(node)
    OPTIONAL MATCH (d:Document)-[:CONTAINS]->(s)

    WITH node, score,
         coalesce(d.title, "") AS doc_title,
         coalesce(d.daod_number, "") AS daod_number,
         coalesce(s.title, node.section, "") AS section_title,
         coalesce(node.text, "") AS text

    RETURN
      node.id AS chunk_id,
      text AS text,
      score AS vec_score,
      doc_title AS doc_title,
      daod_number AS daod_number,
      section_title AS section_title
    ORDER BY vec_score DESC
    LIMIT $limit
    """

    with driver.session() as session:
        return session.run(
            cypher,
            index_name=VECTOR_INDEX,
            k=int(limit),
            embedding=emb,
            limit=int(limit),
        ).data()


def _fulltext_candidates(question: str, limit: int) -> List[Dict[str, Any]]:
    cypher = """
    CALL db.index.fulltext.queryNodes($index_name, $q) YIELD node, score
    WITH node, score
    WHERE node:Chunk

    OPTIONAL MATCH (s:Section)-[:CONTAINS]->(node)
    OPTIONAL MATCH (d:Document)-[:CONTAINS]->(s)

    WITH node, score,
         coalesce(d.title, "") AS doc_title,
         coalesce(d.daod_number, "") AS daod_number,
         coalesce(s.title, node.section, "") AS section_title,
         coalesce(node.text, "") AS text

    RETURN
      node.id AS chunk_id,
      text AS text,
      score AS ft_score,
      doc_title AS doc_title,
      daod_number AS daod_number,
      section_title AS section_title
    ORDER BY ft_score DESC
    LIMIT $limit
    """

    with driver.session() as session:
        return session.run(
            cypher,
            index_name=FULLTEXT_INDEX,
            q=question,
            limit=int(limit),
        ).data()


def retrieve_hybrid(question: str, top_k: int = 8) -> Dict[str, Any]:
    q = (question or "").strip()
    if not q:
        return {"retriever": "hybrid", "question": question, "top_k": top_k, "results": []}

    vec = _vector_candidates(q, limit=VEC_CANDIDATES)
    ft = _fulltext_candidates(q, limit=FT_CANDIDATES)

    merged: Dict[str, Dict[str, Any]] = {}
    for r in vec:
        merged[r["chunk_id"]] = {
            "chunk_id": r["chunk_id"],
            "text": r.get("text", ""),
            "doc_title": r.get("doc_title", ""),
            "daod_number": r.get("daod_number", ""),
            "section_title": r.get("section_title", ""),
            "vec_score": float(r.get("vec_score", 0.0)),
            "ft_score": 0.0,
        }

    for r in ft:
        cid = r["chunk_id"]
        if cid not in merged:
            merged[cid] = {
                "chunk_id": cid,
                "text": r.get("text", ""),
                "doc_title": r.get("doc_title", ""),
                "daod_number": r.get("daod_number", ""),
                "section_title": r.get("section_title", ""),
                "vec_score": 0.0,
                "ft_score": float(r.get("ft_score", 0.0)),
            }
        else:
            merged[cid]["ft_score"] = max(merged[cid]["ft_score"], float(r.get("ft_score", 0.0)))

    ranked = sorted(merged.values(), key=lambda x: (x["vec_score"] + x["ft_score"]), reverse=True)[: int(top_k)]

    return {"retriever": "hybrid", "question": question, "top_k": int(top_k), "results": ranked}


if __name__ == "__main__":
    q = input("Question> ").strip()
    print(retrieve_hybrid(q, top_k=8))
