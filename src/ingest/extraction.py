# src/extraction/extract_entities.py
import json
import os
import re
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from neo4j import GraphDatabase


# ===== Config (your env var style) =====
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma3:12b")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

FETCH_LIMIT = int(os.environ.get("FETCH_LIMIT", "30"))      # how many chunks to pull per loop
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "5"))         # how many chunks to send to LLM per loop (small!)
TIMEOUT_S = int(os.environ.get("OLLAMA_TIMEOUT", "180"))

# extraction “done” marker on Chunk:
DONE_FIELD = "extracted_at"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def normalize_entity_type(t: str) -> str:
    t = clean_ws(t)
    if not t:
        return "Concept"
    # Keep simple & consistent
    t_low = t.lower()
    if t_low in {"person", "people"}:
        return "Person"
    if t_low in {"org", "organization", "organisation"}:
        return "Organization"
    if t_low in {"location", "place"}:
        return "Location"
    if t_low in {"date", "time"}:
        return "Date"
    return t[0].upper() + t[1:]


def extract_json_object(text: str) -> Optional[dict]:
    """
    Pull strict JSON out of model output.
    Handles:
      - pure JSON
      - ```json ... ```
      - extra text around JSON
    """
    if not text:
        return None
    t = text.strip()

    m = re.search(r"```json\s*(\{.*?\})\s*```", t, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    m = re.search(r"(\{.*\})", t, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None

    return None


def build_prompt(chunk_text: str) -> str:
    # You asked for the exact template style. Keep it strict.
    return f"""SYSTEM: You are an information extraction assistant. Return strict JSON only.

USER: Given the text below, extract:
- entities: {{name, type, aliases?}}
- dates: {{label, value, certainty}}
- relationships: {{source_entity, relation_type, target_entity, evidence_excerpt}}
- significance: 2-3 sentences summary

TEXT:
{chunk_text}

Return:
{{
  "entities": [],
  "dates": [],
  "relationships": [],
  "significance": ""
}}
"""


def ollama_generate(prompt: str) -> str:
    """
    Ollama generate API (works for Gemma).
    """
    resp = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        },
        timeout=TIMEOUT_S,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "")


def coerce_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def main() -> None:
    if not NEO4J_PASSWORD:
        raise RuntimeError("Missing NEO4J_PASSWORD (export it first).")

    # Quick Ollama check (fails fast if Ollama is down)
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_HOST}. "
            f"Original error: {type(e).__name__}: {e}"
        )

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()

    # Fetch chunks not yet extracted (works with your Document->Section->Chunk graph)
    fetch_q = f"""
    MATCH (d:Document)-[:HAS_SECTION]->(:Section)-[:CONTAINS]->(c:Chunk)
    WHERE c.{DONE_FIELD} IS NULL
    RETURN d.id AS doc_id, c.id AS chunk_id, c.text AS text
    LIMIT $limit
    """

    # Upsert Entities + connect Chunk->Entity + store chunk significance
    # Your Entity schema: {id, name, type, aliases[], source_doc_id}
    write_entities_q = f"""
    UNWIND $rows AS row
    MATCH (d:Document {{id: row.doc_id}})
    MATCH (c:Chunk {{id: row.chunk_id}})

    SET c.significance = row.significance,
        c.{DONE_FIELD} = $extracted_at,
        c.extract_model = $model

    WITH d, c, row
    UNWIND row.entities AS ent
        MERGE (e:Entity {{id: ent.id}})
        SET e.name = ent.name,
            e.type = ent.type,
            e.aliases = ent.aliases,
            e.source_doc_id = row.doc_id
        MERGE (c)-[m:MENTIONS]->(e)
        SET m.evidence_excerpt = ent.evidence_excerpt
    """

    # Optional: store relationships between entities
    # We keep a single relationship type and put the relation string in a property
    write_rels_q = """
    UNWIND $rels AS r
    MATCH (e1:Entity {id: r.source_id})
    MATCH (e2:Entity {id: r.target_id})
    MERGE (e1)-[x:RELATED_TO {relation_type: r.relation_type}]->(e2)
    SET x.evidence_excerpt = r.evidence_excerpt
    """

    # Optional: store extracted dates on the chunk (simple & avoids extra node type)
    write_dates_q = """
    UNWIND $dates AS drow
    MATCH (c:Chunk {id: drow.chunk_id})
    SET c.extracted_dates = drow.dates
    """

    total = 0

    while True:
        records, _, _ = driver.execute_query(fetch_q, limit=FETCH_LIMIT)
        if not records:
            break

        # Make small LLM batches
        # records is list-like; convert to python list
        recs = list(records)

        rows_to_write: List[Dict[str, Any]] = []
        rels_to_write: List[Dict[str, Any]] = []
        dates_to_write: List[Dict[str, Any]] = []

        # Process in small groups to avoid long stalls
        for start in range(0, len(recs), BATCH_SIZE):
            group = recs[start : start + BATCH_SIZE]

            for rec in group:
                doc_id = rec["doc_id"]
                chunk_id = rec["chunk_id"]
                chunk_text = clean_ws(rec["text"])

                if not chunk_text:
                    continue

                prompt = build_prompt(chunk_text)

                parsed: Optional[dict] = None
                last = ""
                for _ in range(3):  # retry for strict JSON
                    out = ollama_generate(prompt)
                    last = out
                    parsed = extract_json_object(out)
                    if isinstance(parsed, dict):
                        break

                if not isinstance(parsed, dict):
                    print(f"[SKIP] bad JSON for chunk={chunk_id} :: {last[:120]}...")
                    continue

                entities_in = coerce_list(parsed.get("entities"))
                rels_in = coerce_list(parsed.get("relationships"))
                dates_in = coerce_list(parsed.get("dates"))
                significance = clean_ws(parsed.get("significance"))

                # Build entity rows (deterministic id)
                entities_out: List[Dict[str, Any]] = []
                for e in entities_in:
                    if not isinstance(e, dict):
                        continue
                    name = clean_ws(e.get("name"))
                    if not name:
                        continue
                    etype = normalize_entity_type(e.get("type"))
                    aliases = e.get("aliases")
                    if not isinstance(aliases, list):
                        aliases = []
                    aliases = [clean_ws(a) for a in aliases if clean_ws(a)]

                    # deterministic entity id (doc-scoped)
                    ent_id = sha16(f"{doc_id}|{etype}|{name.lower()}")

                    evidence = clean_ws(e.get("evidence_excerpt"))  # optional
                    entities_out.append(
                        {
                            "id": ent_id,
                            "name": name,
                            "type": etype,
                            "aliases": aliases,
                            "evidence_excerpt": evidence,
                        }
                    )

                # Dates: store on chunk as a list of dicts (label/value/certainty)
                cleaned_dates: List[Dict[str, Any]] = []
                for d in dates_in:
                    if not isinstance(d, dict):
                        continue
                    label = clean_ws(d.get("label"))
                    value = clean_ws(d.get("value"))
                    certainty = clean_ws(d.get("certainty"))
                    if label or value:
                        cleaned_dates.append({"label": label, "value": value, "certainty": certainty})
                if cleaned_dates:
                    dates_to_write.append({"chunk_id": chunk_id, "dates": cleaned_dates})

                # Relationships: map names -> ids with same rule
                # If types not given, default to Concept
                def ent_id_for(name: str, typ: str) -> str:
                    return sha16(f"{doc_id}|{normalize_entity_type(typ)}|{name.lower()}")

                for r in rels_in:
                    if not isinstance(r, dict):
                        continue
                    s = clean_ws(r.get("source_entity"))
                    t = clean_ws(r.get("target_entity"))
                    rel_type = clean_ws(r.get("relation_type")) or "RELATED_TO"
                    ev = clean_ws(r.get("evidence_excerpt"))
                    if not s or not t:
                        continue
                    s_type = r.get("source_type") or "Concept"
                    t_type = r.get("target_type") or "Concept"
                    rels_to_write.append(
                        {
                            "source_id": ent_id_for(s, s_type),
                            "target_id": ent_id_for(t, t_type),
                            "relation_type": rel_type,
                            "evidence_excerpt": ev,
                        }
                    )

                rows_to_write.append(
                    {
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "entities": entities_out,
                        "significance": significance,
                    }
                )

            # Write this group
            if rows_to_write:
                driver.execute_query(
                    write_entities_q,
                    rows=rows_to_write,
                    extracted_at=now_iso(),
                    model=LLM_MODEL,
                )
                total += len(rows_to_write)
                print("Chunks extracted + stored:", total)
                rows_to_write = []

            if dates_to_write:
                driver.execute_query(write_dates_q, dates=dates_to_write)
                dates_to_write = []

            if rels_to_write:
                driver.execute_query(write_rels_q, rels=rels_to_write)
                rels_to_write = []

    driver.close()
    print("Done. Total chunks processed:", total)


if __name__ == "__main__":
    main()
