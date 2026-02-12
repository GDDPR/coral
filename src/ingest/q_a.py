import json
import re
import hashlib
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import os
import requests
from neo4j import GraphDatabase


# ===== Constants (no os.getenv for these) =====
WINDOW_SIZE = 40
TARGET_QA_PER_WINDOW = 10

# Prompt safety: truncate each chunk before sending to LLM
MAX_CHUNK_CHARS = 1800

# LLM retries if JSON invalid / errors
LLM_RETRIES = 3

# Minimum usable QAs to accept a window as "done"
# - non-last windows: require >= 5
# - last window: allow >= 2
MIN_QA_NON_LAST = 5
MIN_QA_LAST = 2

# How many docs fetched per loop
DOC_FETCH_LIMIT = 10

# Progress log file
PROGRESS_LOG = Path("data/phase3b_done_windows.jsonl")

# Document markers
DOC_MODEL_FIELD = "qa_model"
DOC_WINDOWS_FIELD = "qa_windows_done"      # array of ints
DOC_FINISHED_FIELD = "qa_generated_at"     # timestamp when all windows done


# ===== Env / runtime config =====
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma3:12b")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "240"))

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")


# ===== Helpers =====
def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def clean_ws(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()


def log_info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def log_error(msg: str) -> None:
    print(f"[ERROR] {msg}", flush=True)


def ensure_progress_parent() -> None:
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)


def load_done_windows_local() -> Set[Tuple[str, int]]:
    """
    Returns set of (doc_id, window_start) from local progress log.
    """
    done: Set[Tuple[str, int]] = set()
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
                    doc_id = obj.get("doc_id")
                    ws = obj.get("window_start")
                    if isinstance(doc_id, str) and doc_id and isinstance(ws, int):
                        done.add((doc_id, ws))
                except Exception:
                    continue
    except Exception as e:
        log_error(f"Could not read progress log {PROGRESS_LOG}: {type(e).__name__}: {e}")

    return done


def append_done_window(doc_id: str, window_start: int, qa_count: int) -> None:
    ensure_progress_parent()
    rec = {
        "doc_id": doc_id,
        "window_start": int(window_start),
        "done_at": now_iso(),
        "model": LLM_MODEL,
        "qa_count": int(qa_count),
    }
    with PROGRESS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def clamp_span(start: int, end: int, n: int) -> Tuple[int, int]:
    start = max(0, min(int(start), n))
    end = max(0, min(int(end), n))
    if end < start:
        start, end = end, start
    return start, end


# ===== Ollama JSON-only call =====
def ollama_generate_json(prompt: str) -> Optional[Any]:
    """
    Uses /api/chat + format=json to enforce strict JSON output.
    Returns parsed JSON or None.
    """
    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": LLM_MODEL,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.9,
            },
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict generator.\n"
                        "You MUST respond in English.\n"
                        "You MUST output ONLY valid JSON.\n"
                        "No markdown. No backticks. No commentary."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        },
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    text = (data.get("message") or {}).get("content") or ""
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def build_prompt(chunks: List[Dict[str, str]], target_count: int, last_window: bool) -> str:
    """
    Provide chunk texts; evidence spans must be within raw chunk text.
    Generate up to target_count; if last_window=True allow fewer.
    """
    lines: List[str] = []
    for c in chunks:
        cid = c["id"]
        txt = c["text"]
        if len(txt) > MAX_CHUNK_CHARS:
            txt = txt[:MAX_CHUNK_CHARS] + " …"
        lines.append(f"CHUNK {cid}:\n{txt}\n")

    chunk_block = "\n".join(lines)

    count_instruction = (
        f"Create up to {target_count} high-signal Q/A pairs."
        if last_window
        else f"Create exactly {target_count} high-signal Q/A pairs."
    )

    return f"""
{count_instruction}

You must generate the questions yourself. Do NOT ask the user for questions.

Each Q/A MUST include evidence spans inside one or more chunks:
"evidence": [{{"chunk_id":"...","start":123,"end":312}}]

Rules:
- English only.
- Output MUST be a JSON array only (no extra text).
- Only use facts stated in the chunks.
- Evidence spans are 0-based offsets into the EXACT chunk text shown.
- Prefer: responsibilities, requirements, rules, definitions, thresholds, prohibitions.
- Keep questions concise; answers direct and actionable.

CHUNKS:
{chunk_block}

Return JSON array of objects with exactly these keys:
[
  {{
    "question": "...",
    "answer": "...",
    "answer_summary": "...",
    "evidence": [{{"chunk_id":"...","start":0,"end":10}}]
  }}
]
""".strip()


# ===== Neo4j Queries =====
FETCH_DOCS_Q = f"""
MATCH (d:Document)
WHERE d.{DOC_FINISHED_FIELD} IS NULL
RETURN d.id AS doc_id, coalesce(d.{DOC_WINDOWS_FIELD}, []) AS windows_done
LIMIT $limit
"""

FETCH_ALL_CHUNK_IDS_Q = """
MATCH (d:Document {id: $doc_id})-[:HAS_SECTION]->(:Section)-[:CONTAINS]->(c:Chunk)
RETURN c.id AS id
ORDER BY c.id
"""

FETCH_CHUNKS_BY_IDS_Q = """
UNWIND $ids AS cid
MATCH (c:Chunk {id: cid})
RETURN c.id AS id, c.text AS text
ORDER BY c.id
"""

WRITE_WINDOW_QA_Q = f"""
MATCH (d:Document {{id: $doc_id}})
SET d.{DOC_MODEL_FIELD} = $model

WITH d
UNWIND $qas AS qa
    MERGE (q:QAPair {{id: qa.id}})
    SET q.question = qa.question,
        q.answer = qa.answer,
        q.answer_summary = qa.answer_summary,
        q.embedding = qa.embedding
    MERGE (d)-[:HAS_QA]->(q)

    WITH d, q, qa
    UNWIND qa.evidence AS ev
        MATCH (c:Chunk {{id: ev.chunk_id}})
        MERGE (q)-[r:EVIDENCED_BY]->(c)
        SET r.start = ev.start,
            r.end = ev.end

WITH d
SET d.{DOC_WINDOWS_FIELD} =
    CASE
        WHEN $window_start IN coalesce(d.{DOC_WINDOWS_FIELD}, [])
        THEN coalesce(d.{DOC_WINDOWS_FIELD}, [])
        ELSE coalesce(d.{DOC_WINDOWS_FIELD}, []) + $window_start
    END
"""

MARK_DOC_FINISHED_Q = f"""
MATCH (d:Document {{id: $doc_id}})
SET d.{DOC_FINISHED_FIELD} = $done_at
"""


def main() -> None:
    if not NEO4J_PASSWORD:
        raise RuntimeError("Missing NEO4J_PASSWORD (export it first).")

    # Quick Ollama reachability check
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_HOST}. "
            f"Original error: {type(e).__name__}: {e}"
        )

    done_windows_local = load_done_windows_local()
    log_info(f"Progress log: {PROGRESS_LOG} (done windows loaded: {len(done_windows_local)})")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()

    total_docs_finished = 0
    total_windows_done = 0
    total_windows_failed = 0
    total_llm_failed = 0
    total_bad_json = 0

    while True:
        try:
            docs, _, _ = driver.execute_query(FETCH_DOCS_Q, limit=DOC_FETCH_LIMIT)
        except Exception as e:
            log_error(f"Neo4j fetch docs failed: {type(e).__name__}: {e}")
            log_error(traceback.format_exc())
            break

        if not docs:
            break

        for d in docs:
            doc_id = d.get("doc_id")
            if not doc_id:
                continue

            windows_done_db = d.get("windows_done") or []
            windows_done_db = [int(x) for x in windows_done_db if isinstance(x, int)]

            # Fetch all chunk ids for doc
            try:
                chunk_ids_recs, _, _ = driver.execute_query(FETCH_ALL_CHUNK_IDS_Q, doc_id=doc_id)
            except Exception as e:
                log_error(f"Neo4j fetch chunk ids failed (doc={doc_id}): {type(e).__name__}: {e}")
                log_error(traceback.format_exc())
                continue

            all_chunk_ids = [r["id"] for r in chunk_ids_recs if r.get("id")]
            if not all_chunk_ids:
                try:
                    driver.execute_query(MARK_DOC_FINISHED_Q, doc_id=doc_id, done_at=now_iso())
                    total_docs_finished += 1
                    log_info(f"FINISHED doc={doc_id} (no chunks) docs_finished={total_docs_finished}")
                except Exception as e:
                    log_error(f"Mark doc finished failed (doc={doc_id}): {type(e).__name__}: {e}")
                continue

            window_starts = list(range(0, len(all_chunk_ids), WINDOW_SIZE))

            for ws in window_starts:
                # Skip if already done
                if ws in windows_done_db or (doc_id, ws) in done_windows_local:
                    continue

                window_ids = all_chunk_ids[ws : ws + WINDOW_SIZE]
                last_window = (ws + WINDOW_SIZE) >= len(all_chunk_ids)

                # Fetch chunk texts
                try:
                    chunk_recs, _, _ = driver.execute_query(FETCH_CHUNKS_BY_IDS_Q, ids=window_ids)
                except Exception as e:
                    total_windows_failed += 1
                    log_error(f"Neo4j fetch window chunks failed (doc={doc_id}, ws={ws}): {type(e).__name__}: {e}")
                    log_error(traceback.format_exc())
                    continue

                chunks: List[Dict[str, str]] = []
                chunk_text_map: Dict[str, str] = {}

                for r in chunk_recs:
                    cid = r.get("id")
                    txt = clean_ws(r.get("text"))
                    if not cid or not txt:
                        continue
                    chunks.append({"id": cid, "text": txt})
                    chunk_text_map[cid] = txt

                if not chunks:
                    # Mark window done with 0 QAs
                    try:
                        driver.execute_query(
                            WRITE_WINDOW_QA_Q,
                            doc_id=doc_id,
                            qas=[],
                            model=LLM_MODEL,
                            window_start=int(ws),
                        )
                        append_done_window(doc_id, int(ws), 0)
                        done_windows_local.add((doc_id, int(ws)))
                        total_windows_done += 1
                        log_info(f"OK doc={doc_id} window={ws} (no chunk texts) windows_done={total_windows_done}")
                    except Exception as e:
                        total_windows_failed += 1
                        log_error(f"Neo4j mark window done failed (doc={doc_id}, ws={ws}): {type(e).__name__}: {e}")
                        log_error(traceback.format_exc())
                    continue

                # LLM generation
                prompt = build_prompt(chunks, target_count=TARGET_QA_PER_WINDOW, last_window=last_window)

                arr: Optional[list] = None
                last_err: Optional[Exception] = None

                for _ in range(LLM_RETRIES):
                    try:
                        out = ollama_generate_json(prompt)
                        if isinstance(out, list):
                            arr = out
                            break
                    except Exception as e:
                        last_err = e

                if not isinstance(arr, list):
                    total_bad_json += 1
                    total_windows_failed += 1
                    log_error(f"Bad JSON (doc={doc_id}, ws={ws}). model={LLM_MODEL}")
                    if last_err:
                        log_error(f"LLM error: {type(last_err).__name__}: {last_err}")
                    continue

                # Validate / normalize
                qas_out: List[Dict[str, Any]] = []
                seen_q: Set[str] = set()

                for qa in arr:
                    if not isinstance(qa, dict):
                        continue

                    q = clean_ws(qa.get("question"))
                    a = clean_ws(qa.get("answer"))
                    if not q or not a:
                        continue

                    q_key = q.lower()
                    if q_key in seen_q:
                        continue
                    seen_q.add(q_key)

                    summary = clean_ws(qa.get("answer_summary")) or None

                    evidence_in = qa.get("evidence")
                    evidence_list = evidence_in if isinstance(evidence_in, list) else []

                    ev_out: List[Dict[str, Any]] = []
                    for ev in evidence_list:
                        if not isinstance(ev, dict):
                            continue
                        cid = clean_ws(ev.get("chunk_id"))
                        if not cid or cid not in chunk_text_map:
                            continue

                        n = len(chunk_text_map[cid])
                        try:
                            start = int(ev.get("start", 0))
                            end = int(ev.get("end", 0))
                        except Exception:
                            start, end = 0, 0

                        start, end = clamp_span(start, end, n)
                        if end <= start:
                            continue

                        ev_out.append({"chunk_id": cid, "start": start, "end": end})

                    if not ev_out:
                        continue

                    qa_id = sha16(f"{doc_id}|{q.lower()}")

                    qas_out.append(
                        {
                            "id": qa_id,
                            "question": q,
                            "answer": a,
                            "answer_summary": summary,
                            "embedding": [],  # fill later by QA embedding script
                            "evidence": ev_out,
                        }
                    )

                    # cap: up to target per window, but last window can be less
                    if len(qas_out) >= TARGET_QA_PER_WINDOW:
                        break

                # Minimum threshold (last window can be smaller)
                min_required = MIN_QA_LAST if last_window else MIN_QA_NON_LAST
                if len(qas_out) < min_required:
                    total_windows_failed += 1
                    log_error(
                        f"Doc {doc_id} window {ws}: only {len(qas_out)} usable QAs "
                        f"(min required {min_required}, last_window={last_window}). "
                        f"Not marking window done so it can be retried."
                    )
                    continue

                # Write window QAs
                try:
                    driver.execute_query(
                        WRITE_WINDOW_QA_Q,
                        doc_id=doc_id,
                        qas=qas_out,
                        model=LLM_MODEL,
                        window_start=int(ws),
                    )
                    append_done_window(doc_id, int(ws), len(qas_out))
                    done_windows_local.add((doc_id, int(ws)))
                    total_windows_done += 1
                    log_info(f"OK doc={doc_id} window={ws} qas={len(qas_out)} windows_done={total_windows_done}")
                except Exception as e:
                    total_windows_failed += 1
                    log_error(f"Neo4j write window failed (doc={doc_id}, ws={ws}): {type(e).__name__}: {e}")
                    log_error(traceback.format_exc())
                    continue

            # After all windows, mark doc finished if complete
            try:
                recs, _, _ = driver.execute_query(
                    f"MATCH (d:Document {{id:$doc_id}}) RETURN coalesce(d.{DOC_WINDOWS_FIELD}, []) AS w",
                    doc_id=doc_id,
                )
                w_done = recs[0]["w"] if recs else []
                w_done = set(int(x) for x in w_done if isinstance(x, int))
            except Exception:
                w_done = set(windows_done_db)

            if set(window_starts).issubset(w_done):
                try:
                    driver.execute_query(MARK_DOC_FINISHED_Q, doc_id=doc_id, done_at=now_iso())
                    total_docs_finished += 1
                    log_info(f"FINISHED doc={doc_id} docs_finished={total_docs_finished}")
                except Exception as e:
                    log_error(f"Mark doc finished failed (doc={doc_id}): {type(e).__name__}: {e}")

    driver.close()

    print("Done.")
    print(f"  Docs finished:     {total_docs_finished}")
    print(f"  Windows done:      {total_windows_done}")
    print(f"  Windows failed:    {total_windows_failed}")
    print(f"  LLM failures:      {total_llm_failed}")
    print(f"  Bad JSON windows:  {total_bad_json}")
    print(f"  Progress log:      {PROGRESS_LOG.resolve()}")


if __name__ == "__main__":
    main()
