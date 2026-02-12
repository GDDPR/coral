# src/chunking/extraction.py
#
# Phase 3a — LLM-driven extraction (Gemma via Ollama) into Neo4j
#
# What this version guarantees:
# 1) It logs EVERY successfully-processed chunk id to a local file (JSONL),
#    so you can restart and it will skip those chunks immediately.
# 2) It NEVER “marks a chunk done” in Neo4j until ALL writes for that chunk succeed
#    (entities + dates + relationships + significance).
# 3) If an error happens, it logs the chunk id and keeps going.
#
# Neo4j model:
# - Entity {id, name, type, aliases[], source_doc_id}  (unique by e.id)
# - Chunk stores: significance, extracted_at, extract_model
# - Chunk -> Entity edges:
#     (c)-[:MENTIONS {evidence_excerpt}]->(e)      for non-date entities
#     (c)-[:MENTIONS_DATE {label, certainty}]->(e) for Date entities
# - Entity -> Entity edges:
#     (e1)-[:RELATED_TO {relation_type, evidence_excerpt}]->(e2)
#
# Run:
#   export NEO4J_PASSWORD="neo4jneo4j"
#   export OLLAMA_HOST="http://localhost:11434"
#   export LLM_MODEL="gemma3:12b"
#   python3 src/chunking/extraction.py

import json
import os
import re
import hashlib
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from neo4j import GraphDatabase


# ===== Config (env vars) =====
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma3:12b")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

FETCH_LIMIT = int(os.environ.get("FETCH_LIMIT", "50"))      # chunks fetched per loop from Neo4j
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "5"))         # LLM calls per loop
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "180"))

# local progress log (successful chunks)
PROGRESS_LOG = Path(os.environ.get("PHASE3_PROGRESS_LOG", "data/phase3a_done.jsonl"))

# marker fields on Chunk
DONE_FIELD = "extracted_at"
MODEL_FIELD = "extract_model"
SIG_FIELD = "significance"


# ===== Helpers =====
def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def clean_ws(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()


def normalize_entity_type(t: Any) -> str:
    t = clean_ws(t)
    if not t:
        return "Concept"
    tl = t.lower()
    if tl in {"person", "people"}:
        return "Person"
    if tl in {"org", "organization", "organisation"}:
        return "Organization"
    if tl in {"location", "place"}:
        return "Location"
    if tl in {"date", "time"}:
        return "Date"
    if tl in {"concept"}:
        return "Concept"
    return t[0].upper() + t[1:]


def coerce_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def extract_json_object(text: str) -> Optional[dict]:
    """
    Pull a JSON object out of LLM output.
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
    # Your strict JSON template
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
    resp = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        },
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "")


def log_error(msg: str) -> None:
    print(f"[ERROR] {msg}", flush=True)


def log_info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def ensure_progress_log_parent() -> None:
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)


def load_done_chunk_ids() -> Set[str]:
    """
    Reads data/phase3a_done.jsonl and returns chunk ids already completed.
    Each line is JSON like: {"chunk_id":"...", "doc_id":"...", "done_at":"..."}
    """
    done: Set[str] = set()
    if not PROGRESS_LOG.exists():
        return done

    try:
        with PROGRESS_LOG.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    cid = obj.get("chunk_id")
                    if isinstance(cid, str) and cid:
                        done.add(cid)
                except Exception:
                    # ignore corrupted line, keep going
                    continue
    except Exception as e:
        log_error(f"Could not read progress log {PROGRESS_LOG}: {type(e).__name__}: {e}")

    return done


def append_done_chunk(doc_id: str, chunk_id: str) -> None:
    """
    Append a successful chunk to progress log (JSONL).
    """
    ensure_progress_log_parent()
    rec = {"doc_id": doc_id, "chunk_id": chunk_id, "done_at": now_iso(), "model": LLM_MODEL}
    with PROGRESS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ===== Build extraction payload for one chunk =====
def build_entities_payload(doc_id: str, entities_in: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
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
        ent_id = sha16(f"{doc_id}|{etype}|{name.lower()}")
        evidence = clean_ws(e.get("evidence_excerpt"))
        out.append(
            {
                "id": ent_id,
                "name": name,
                "type": etype,
                "aliases": aliases,
                "evidence_excerpt": evidence,
            }
        )
    return out


def build_dates_payload(doc_id: str, dates_in: List[Any]) -> List[Dict[str, Any]]:
    """
    Dates cannot be stored as list-of-maps property in Neo4j.
    We store them as Entity nodes with type='Date' and link via MENTIONS_DATE.
    """
    out: List[Dict[str, Any]] = []
    for d in dates_in:
        if not isinstance(d, dict):
            continue
        label = clean_ws(d.get("label"))
        value = clean_ws(d.get("value"))
        certainty = clean_ws(d.get("certainty"))
        if not value:
            continue
        date_ent_id = sha16(f"{doc_id}|Date|{value.lower()}")
        out.append({"id": date_ent_id, "value": value, "label": label, "certainty": certainty})
    return out


def build_rels_payload(doc_id: str, rels_in: List[Any]) -> List[Dict[str, Any]]:
    """
    Relationships reference source_entity and target_entity names.
    We map those names to deterministic ids using same rule as entities.
    If model doesn't provide types, default to Concept.
    """
    out: List[Dict[str, Any]] = []

    def ent_id_for(name: str, typ: Any) -> str:
        return sha16(f"{doc_id}|{normalize_entity_type(typ)}|{name.lower()}")

    for r in rels_in:
        if not isinstance(r, dict):
            continue
        s = clean_ws(r.get("source_entity"))
        t = clean_ws(r.get("target_entity"))
        rel_type = clean_ws(r.get("relation_type")) or "RELATED_TO"
        evidence = clean_ws(r.get("evidence_excerpt"))
        if not s or not t:
            continue
        s_type = r.get("source_type") or "Concept"
        t_type = r.get("target_type") or "Concept"
        out.append(
            {
                "source_id": ent_id_for(s, s_type),
                "target_id": ent_id_for(t, t_type),
                "relation_type": rel_type,
                "evidence_excerpt": evidence,
            }
        )
    return out


# ===== Neo4j write (single-chunk atomic write) =====
WRITE_ONE_CHUNK_Q = f"""
// One chunk, one transaction, atomic "done" marker
MATCH (d:Document {{id: $doc_id}})
MATCH (c:Chunk {{id: $chunk_id}})

SET c.{SIG_FIELD} = $significance,
    c.{MODEL_FIELD} = $model,
    c.{DONE_FIELD} = $extracted_at

WITH d, c

// ---- Non-date entities ----
UNWIND $entities AS ent
MERGE (e:Entity {{id: ent.id}})
SET e.name = ent.name,
    e.type = ent.type,
    e.aliases = ent.aliases,
    e.source_doc_id = d.id
MERGE (c)-[m:MENTIONS]->(e)
SET m.evidence_excerpt = ent.evidence_excerpt

WITH d, c

// ---- Date entities ----
UNWIND $dates AS dt
MERGE (de:Entity {{id: dt.id}})
SET de.name = dt.value,
    de.type = 'Date',
    de.aliases = [],
    de.source_doc_id = d.id
MERGE (c)-[md:MENTIONS_DATE]->(de)
SET md.label = coalesce(dt.label,''),
    md.certainty = coalesce(dt.certainty,'')

WITH d, c

// ---- Relationships ----
UNWIND $rels AS r
MATCH (e1:Entity {{id: r.source_id}})
MATCH (e2:Entity {{id: r.target_id}})
MERGE (e1)-[x:RELATED_TO {{relation_type: r.relation_type}}]->(e2)
SET x.evidence_excerpt = r.evidence_excerpt
"""


def main() -> None:
    if not NEO4J_PASSWORD:
        raise RuntimeError("Missing NEO4J_PASSWORD (export it first).")

    # Ollama quick check
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_HOST}. "
            f"Original error: {type(e).__name__}: {e}"
        )

    done_local = load_done_chunk_ids()
    log_info(f"Progress log: {PROGRESS_LOG} (done chunks loaded: {len(done_local)})")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()

    # Fetch chunks not yet extracted from Neo4j.
    # We ALSO locally skip any that are in the progress log.
    fetch_q = f"""
    MATCH (d:Document)-[:HAS_SECTION]->(:Section)-[:CONTAINS]->(c:Chunk)
    WHERE c.{DONE_FIELD} IS NULL
    RETURN d.id AS doc_id, c.id AS chunk_id, c.text AS text
    LIMIT $limit
    """

    total_ok = 0
    total_skipped_local = 0
    total_llm_fail = 0
    total_bad_json = 0
    total_write_fail = 0

    while True:
        try:
            records, _, _ = driver.execute_query(fetch_q, limit=FETCH_LIMIT)
        except Exception as e:
            log_error(f"Neo4j fetch failed: {type(e).__name__}: {e}")
            log_error(traceback.format_exc())
            break

        if not records:
            break

        recs = list(records)

        # local-skip filter
        filtered: List[Tuple[str, str, str]] = []
        for rec in recs:
            doc_id = rec["doc_id"]
            chunk_id = rec["chunk_id"]
            text = rec["text"] or ""
            if chunk_id in done_local:
                total_skipped_local += 1
                continue
            filtered.append((doc_id, chunk_id, text))

        if not filtered:
            # If everything fetched was already done locally, continue fetching next page/loop.
            log_info(f"Fetched {len(recs)} chunks, but all were already in progress log; continuing...")
            continue

        # Process in small batches (LLM calls)
        for start in range(0, len(filtered), BATCH_SIZE):
            group = filtered[start : start + BATCH_SIZE]

            for doc_id, chunk_id, chunk_text_raw in group:
                chunk_text = clean_ws(chunk_text_raw)

                # If chunk text is empty, we can mark as done with empty extraction
                # (still does an atomic write, no entities/dates/rels).
                if not chunk_text:
                    try:
                        driver.execute_query(
                            WRITE_ONE_CHUNK_Q,
                            doc_id=doc_id,
                            chunk_id=chunk_id,
                            significance="",
                            entities=[],
                            dates=[],
                            rels=[],
                            model=LLM_MODEL,
                            extracted_at=now_iso(),
                        )
                        append_done_chunk(doc_id, chunk_id)
                        done_local.add(chunk_id)
                        total_ok += 1
                        log_info(f"OK (empty) chunk={chunk_id} total_ok={total_ok}")
                    except Exception as e:
                        total_write_fail += 1
                        log_error(f"Neo4j write failed (chunk={chunk_id}): {type(e).__name__}: {e}")
                        log_error(traceback.format_exc())
                    continue

                # ---- LLM extraction (with retries) ----
                prompt = build_prompt(chunk_text)
                parsed: Optional[dict] = None
                last = ""

                try:
                    for _ in range(3):
                        out = ollama_generate(prompt)
                        last = out
                        parsed = extract_json_object(out)
                        if isinstance(parsed, dict):
                            break
                except Exception as e:
                    total_llm_fail += 1
                    log_error(f"LLM call failed (chunk={chunk_id}): {type(e).__name__}: {e}")
                    # don't mark done; keep going
                    continue

                if not isinstance(parsed, dict):
                    total_bad_json += 1
                    log_error(f"Bad JSON (chunk={chunk_id}) :: {last[:250]}...")
                    # don't mark done; keep going
                    continue

                entities_in = coerce_list(parsed.get("entities"))
                dates_in = coerce_list(parsed.get("dates"))
                rels_in = coerce_list(parsed.get("relationships"))
                significance = clean_ws(parsed.get("significance"))

                # ---- Build deterministic payloads ----
                entities = build_entities_payload(doc_id, entities_in)
                dates = build_dates_payload(doc_id, dates_in)
                rels = build_rels_payload(doc_id, rels_in)

                # ---- Atomic Neo4j write for this chunk ----
                try:
                    driver.execute_query(
                        WRITE_ONE_CHUNK_Q,
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        significance=significance,
                        entities=entities,
                        dates=dates,
                        rels=rels,
                        model=LLM_MODEL,
                        extracted_at=now_iso(),
                    )
                    append_done_chunk(doc_id, chunk_id)
                    done_local.add(chunk_id)
                    total_ok += 1
                    log_info(f"OK chunk={chunk_id} entities={len(entities)} dates={len(dates)} rels={len(rels)} total_ok={total_ok}")
                except Exception as e:
                    total_write_fail += 1
                    log_error(f"Neo4j write failed (chunk={chunk_id}): {type(e).__name__}: {e}")
                    log_error(traceback.format_exc())
                    # don't mark done; keep going
                    continue

    driver.close()

    print("Done.")
    print(f"  OK chunks:                 {total_ok}")
    print(f"  Locally skipped chunks:    {total_skipped_local}")
    print(f"  LLM failures:              {total_llm_fail}")
    print(f"  Bad JSON responses:        {total_bad_json}")
    print(f"  Neo4j write failures:      {total_write_fail}")
    print(f"  Progress log:              {PROGRESS_LOG.resolve()}")


if __name__ == "__main__":
    main()
