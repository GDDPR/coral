import json
import os
import re
import sys
from typing import Any, Dict, List

import requests
from neo4j import GraphDatabase

# ---- Neo4j ----
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

# ---- Ollama ----
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")

HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "60"))

# ---- Retrieval params ----
TOPK = int(os.getenv("TOPK", "8"))
TOPK_VEC = int(os.getenv("TOPK_VEC", str(TOPK)))
TOPK_FT = int(os.getenv("TOPK_FT", str(TOPK)))
ALPHA = float(os.getenv("ALPHA", "0.6"))
BETA = float(os.getenv("BETA", "0.4"))

CHUNK_VECTOR_INDEX = "chunkembedding"
FULLTEXT_INDEX = "fulltextcontent"


def clean_ws(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()


def ollama_embed(text: str) -> List[float]:
    r = requests.post(
        f"{OLLAMA_HOST}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=HTTP_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    if "embeddings" in data and isinstance(data["embeddings"], list) and data["embeddings"]:
        return data["embeddings"][0]
    if "embedding" in data and isinstance(data["embedding"], list):
        return data["embedding"]
    raise RuntimeError(f"Unexpected embed response keys: {list(data.keys())}")


VEC_Q = """
CALL db.index.vector.queryNodes($index, $k, $vec)
YIELD node, score
RETURN node.id AS chunk_id, node.text AS text, score AS score
"""

FT_Q = """
CALL db.index.fulltext.queryNodes($index, $q)
YIELD node, score
WHERE node:Chunk
RETURN node.id AS chunk_id, node.text AS text, score AS score
LIMIT $k
"""


def minmax(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo <= 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def retrieve(question: str) -> Dict[str, Any]:
    qvec = ollama_embed(question)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    vec_recs, _, _ = driver.execute_query(VEC_Q, index=CHUNK_VECTOR_INDEX, k=TOPK_VEC, vec=qvec)
    ft_recs, _, _ = driver.execute_query(FT_Q, index=FULLTEXT_INDEX, q=question, k=TOPK_FT)
    driver.close()

    v_scores: Dict[str, float] = {}
    v_text: Dict[str, str] = {}
    for r in vec_recs:
        cid = clean_ws(r.get("chunk_id"))
        if cid:
            v_scores[cid] = float(r.get("score") or 0.0)
            v_text[cid] = r.get("text") or ""

    ft_scores: Dict[str, float] = {}
    ft_text: Dict[str, str] = {}
    for r in ft_recs:
        cid = clean_ws(r.get("chunk_id"))
        if cid:
            ft_scores[cid] = float(r.get("score") or 0.0)
            ft_text[cid] = r.get("text") or ""

    v_norm = minmax(v_scores)
    ft_norm = minmax(ft_scores)

    all_ids = set(v_norm.keys()) | set(ft_norm.keys())

    hits: List[Dict[str, Any]] = []
    for cid in all_ids:
        score = ALPHA * v_norm.get(cid, 0.0) + BETA * ft_norm.get(cid, 0.0)
        hits.append(
            {
                "chunk_id": cid,
                "score": score,
                "score_vec": v_norm.get(cid, 0.0),
                "score_ft": ft_norm.get(cid, 0.0),
                "text": v_text.get(cid) or ft_text.get(cid) or "",
            }
        )

    hits.sort(key=lambda x: x["score"], reverse=True)
    hits = hits[:TOPK]

    return {"query": question, "retrieval": {"chunks": hits}}


def main() -> None:
    if len(sys.argv) >= 2:
        question = clean_ws(" ".join(sys.argv[1:]))
    else:
        question = clean_ws(input("Question> "))

    out = retrieve(question)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
