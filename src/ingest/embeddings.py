import os
from datetime import datetime, timezone

import ollama
from neo4j import GraphDatabase


EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text:latest")


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def embed(texts):
    resp = ollama.embed(model=EMBED_MODEL, input=texts)
    return resp["embeddings"]


def main():
    URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    USER = os.environ.get("NEO4J_USER", "neo4j")
    PWD = os.environ.get("NEO4J_PASSWORD")
    if not PWD:
        raise RuntimeError("Missing NEO4J_PASSWORD (export it first).")

    os.environ.setdefault("OLLAMA_HOST", os.environ.get("OLLAMA_HOST", "http://localhost:11434"))

    FETCH_LIMIT = int(os.environ.get("FETCH_LIMIT", "200"))
    BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "16"))

    print("Neo4j:", URI)
    print("Ollama:", os.environ["OLLAMA_HOST"])
    print("Model:", EMBED_MODEL)
    print("Fetch limit:", FETCH_LIMIT)
    print("Batch size:", BATCH_SIZE)

    driver = GraphDatabase.driver(URI, auth=(USER, PWD))
    driver.verify_connectivity()

    # ✅ Chunk schema
    driver.execute_query("""
    CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
    FOR (c:Chunk)
    REQUIRE c.id IS UNIQUE
    """)
    print("Constraint ensured.")

    fetch_q = """
    MATCH (c:Chunk)
    WHERE c.embedding IS NULL OR size(c.embedding) = 0
    RETURN c.id AS id, c.text AS text
    LIMIT $limit
    """

    write_q = """
    UNWIND $rows AS row
    MATCH (c:Chunk {id: row.id})
    SET c.embedding = row.embedding,
        c.embedding_model = $model,
        c.embedded_at = $embedded_at
    """

    total = 0

    while True:
        records, _, _ = driver.execute_query(fetch_q, limit=FETCH_LIMIT)
        if not records:
            break

        rows = []
        for r in records:
            text = (r["text"] or "").strip()
            if not text:
                continue
            rows.append({"id": r["id"], "text": text})

        if not rows:
            break

        to_write = []

        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            texts = [b["text"] for b in batch]

            vectors = embed(texts)

            if len(vectors) != len(texts):
                raise RuntimeError(
                    f"Ollama returned {len(vectors)} embeddings for {len(texts)} texts. "
                    f"Try smaller EMBED_BATCH_SIZE."
                )

            for b, v in zip(batch, vectors):
                to_write.append({"id": b["id"], "embedding": v})

        driver.execute_query(
            write_q,
            rows=to_write,
            model=EMBED_MODEL,
            embedded_at=now_iso(),
        )

        total += len(to_write)
        print("Embedded + stored:", total)

    driver.close()
    print("Done. Total chunks embedded:", total)


if __name__ == "__main__":
    main()
