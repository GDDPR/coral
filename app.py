# app.py
import os
import re
from typing import Any, Dict, List

import requests
import streamlit as st
from neo4j import GraphDatabase

from src.retrieve.question_rewrite import rewrite_question
from src.retrieve.expand_section import expand_to_section
from src.retrieve.retrieve_keyword_hybrid import retrieve_keyword_hybrid  # default for now

# ---- Ollama ----
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:12b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))

# ---- Defaults ----
TOPK_DEFAULT = int(os.getenv("TOPK", "8"))
MAX_CHARS_PER_CHUNK = int(os.getenv("MAX_CHARS_PER_CHUNK", "1600"))
SIBLING_WINDOW = int(os.getenv("SIBLING_WINDOW", "2"))

# Collection info (optional; set later)
DAOD6000_DOC_COUNT = os.getenv("DAOD6000_DOC_COUNT", "___")

# ---- Neo4j (for metadata hydration) ----
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def clean_ws(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()


def hydrate_doc_meta(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensures every chunk dict has:
      - doc_title
      - daod_number
      - section_title

    This fixes the "only section shows" problem when expand_to_section()
    returns chunks without document metadata.
    """
    if not chunks:
        return []

    ids = [clean_ws(c.get("chunk_id")) for c in chunks if clean_ws(c.get("chunk_id"))]
    if not ids:
        return chunks

    cypher = """
    UNWIND $ids AS cid
    MATCH (c:Chunk {id: cid})
    OPTIONAL MATCH (s:Section)-[:CONTAINS]-(c)
    OPTIONAL MATCH (d:Document)-[:HAS_SECTION]-(s)
    WITH c,
         head(collect(DISTINCT d.title)) AS doc_title,
         head(collect(DISTINCT d.daod_number)) AS daod_number,
         head(collect(DISTINCT s.title)) AS section_title
    RETURN
      c.id AS chunk_id,
      coalesce(doc_title, "") AS doc_title,
      coalesce(daod_number, "") AS daod_number,
      coalesce(section_title, c.section, "") AS section_title
    """

    with driver.session() as session:
        rows = session.run(cypher, ids=ids).data()

    meta_by_id = {r["chunk_id"]: r for r in rows if r.get("chunk_id")}

    out: List[Dict[str, Any]] = []
    for ch in chunks:
        cid = clean_ws(ch.get("chunk_id"))
        meta = meta_by_id.get(cid, {})
        # only fill missing
        if not clean_ws(ch.get("doc_title")) and meta.get("doc_title"):
            ch["doc_title"] = meta["doc_title"]
        if not clean_ws(ch.get("daod_number")) and meta.get("daod_number"):
            ch["daod_number"] = meta["daod_number"]
        if not clean_ws(ch.get("section_title")) and meta.get("section_title"):
            ch["section_title"] = meta["section_title"]
        out.append(ch)

    return out


def build_context(chunks: List[Dict[str, Any]], top_k: int) -> str:
    blocks: List[str] = []
    for ch in (chunks or [])[:top_k]:
        cid = clean_ws(ch.get("chunk_id"))
        txt = clean_ws(ch.get("text"))

        doc_title = clean_ws(ch.get("doc_title"))
        daod_num = clean_ws(ch.get("daod_number"))
        section_title = clean_ws(ch.get("section_title"))

        if not cid or not txt:
            continue

        if len(txt) > MAX_CHARS_PER_CHUNK:
            txt = txt[:MAX_CHARS_PER_CHUNK].rstrip()

        header_parts = []
        # Prefer showing the actual document title clearly
        if doc_title:
            header_parts.append(doc_title)
        elif daod_num:
            header_parts.append(f"DAOD {daod_num}")

        if section_title:
            header_parts.append(f"Section: {section_title}")

        header = " — ".join(header_parts) if header_parts else "DAOD 6000"
        blocks.append(f"{header}\n[chunk_id={cid}]\n{txt}\n")

    return "\n".join(blocks).strip()


def build_prompt(question: str, context: str) -> str:
    refusal = "I have searched the content of my DAOD 6000 database and found no matching information — I cannot answer."

    return f"""SYSTEM:
You are a retrieval-based assistant for the Defence Administrative Orders and Directives (DAOD), and only for the 6000 series that are focused on Information Management. The series or collection contains {DAOD6000_DOC_COUNT} documents.

You MUST answer only using the provided context.
If the answer is not contained in the context, say exactly:
"{refusal}"

Do NOT use prior knowledge.
Do NOT speculate.

QUESTION:
{question}

CONTEXT:
{context}

INSTRUCTIONS:
- Use ONLY the context above.
- If missing, say you cannot answer (exact sentence required).
- Cite which section supports your answer.
- Include the chunk_id(s) you used.

Return format:
Answer:
...

Citations:
- <doc title> — <section title> — <chunk_id>
"""


def ollama_generate(prompt: str) -> str:
    r = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        },
        timeout=OLLAMA_TIMEOUT,
    )
    r.raise_for_status()
    return r.json().get("response", "")


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="CORAL Ask", layout="wide")
st.title("CORAL Ask (Neo4j + Ollama)")

k = st.number_input("top_k (chunks)", min_value=1, max_value=50, value=TOPK_DEFAULT, step=1)
q = st.text_input("Question", placeholder="Ask a question about DAOD 6000-series (Information Management)...")

if st.button("Ask") and q.strip():
    user_q = clean_ws(q)

    with st.spinner("Rewriting question…"):
        rq = rewrite_question(user_q)
        clean_query = clean_ws(rq.get("clean_query", user_q))

    with st.spinner("Retrieving (keyword+vector hybrid)…"):
        retrieval = retrieve_keyword_hybrid(clean_query, top_k=int(k))
        seed_chunks = retrieval.get("results", [])

    with st.spinner("Expanding to section context…"):
        expanded_chunks = expand_to_section(seed_chunks, window=SIBLING_WINDOW)
        final_chunks = expanded_chunks or seed_chunks

    # ✅ CRITICAL FIX: make sure expanded chunks have doc_title/daod_number/section_title
    with st.spinner("Hydrating document titles…"):
        final_chunks = hydrate_doc_meta(final_chunks)

    context = build_context(final_chunks, top_k=int(k))
    prompt = build_prompt(user_q, context)

    with st.spinner("Generating answer…"):
        answer = ollama_generate(prompt).strip()

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Retrieved chunks (debug)"):
        st.json(final_chunks)

    with st.expander("DEBUG: seed vs expanded chunk keys"):
        st.write("Seed chunk keys:")
        if seed_chunks:
            st.write(list(seed_chunks[0].keys()))
            st.json(seed_chunks[0])
        else:
            st.write("No seed chunks")

        st.write("Expanded chunk keys:")
        if expanded_chunks:
            st.write(list(expanded_chunks[0].keys()))
            st.json(expanded_chunks[0])
        else:
            st.write("No expanded chunks")

    with st.expander("DEBUG: context sent to LLM"):
        st.text(context)

