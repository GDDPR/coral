# app.py
import os
import re
import time
from typing import Any, Dict, List, Callable

import requests
import streamlit as st
from neo4j import GraphDatabase

from src.retrieve.question_rewrite import rewrite_question
from src.retrieve.expand_section import expand_to_section

from src.retrieve.retrieve_keyword import retrieve_keyword
from src.retrieve.retrieve_entity import retrieve_entity
from src.retrieve.retrieve_hybrid import retrieve_hybrid
from src.retrieve.retrieve_keyword_hybrid import retrieve_keyword_hybrid  # default

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

# ---- Retriever registry ----
RETRIEVERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "keyword": retrieve_keyword,
    "entity": retrieve_entity,
    "hybrid": retrieve_hybrid,
    "keyword_hybrid": retrieve_keyword_hybrid,
}


def clean_ws(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()


def hydrate_doc_meta(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensures every chunk dict has:
      - doc_title
      - daod_number
      - section_title
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

        header_parts: List[str] = []
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


def run_retrieval(retriever_name: str, query: str, top_k: int) -> Dict[str, Any]:
    fn = RETRIEVERS.get(retriever_name)
    if fn is None:
        raise ValueError(f"Unknown retriever '{retriever_name}'")
    data = fn(query, top_k=top_k)
    if isinstance(data, dict) and not clean_ws(data.get("question")):
        data["question"] = query
    return data


def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _run_one_retriever(name: str, user_q: str, clean_query: str, top_k: int) -> Dict[str, Any]:
    """
    Runs: retrieval -> expand_to_section -> hydrate_doc_meta -> context -> prompt -> ollama
    Returns: dict with prompt, answer, and debug fields.
    """
    retrieval = run_retrieval(name, clean_query, top_k=top_k)
    seed_chunks = retrieval.get("results", [])

    expanded_chunks = expand_to_section(seed_chunks, window=SIBLING_WINDOW)
    final_chunks = expanded_chunks or seed_chunks

    final_chunks = hydrate_doc_meta(final_chunks)

    context = build_context(final_chunks, top_k=top_k)
    prompt = build_prompt(user_q, context)

    answer = ollama_generate(prompt).strip()

    return {
        "prompt": prompt,
        "answer": answer,
        "chunks": final_chunks,
        "seed_chunks": seed_chunks,
        "expanded_chunks": expanded_chunks,
        "context": context,
    }


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="CORAL Ask", layout="wide")
st.title("CORAL Ask (Neo4j + Ollama)")

# ---- session history init ----
if "history" not in st.session_state:
    # each item: {"ts":..., "question":..., "clean_query":..., "mode":..., "prompts":{...}, "answers":{...}}
    st.session_state.history = []

with st.sidebar:
    st.header("Retrieval Settings")

    retriever_choice = st.selectbox(
        "Retriever",
        options=["keyword_hybrid", "hybrid", "keyword", "entity", "all_retrievers"],
        index=0,
        help="Choose one retriever, or run all four for comparison.",
    )

    k = st.number_input("top_k (chunks)", min_value=1, max_value=50, value=TOPK_DEFAULT, step=1)

    st.caption(f"Ollama model: {LLM_MODEL}")

    st.divider()

    colA, colB = st.columns([3, 2])
    with colA:
        st.subheader("Your chats")
    with colB:
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    for item in reversed(st.session_state.history):
        title = f"{item['ts']} — {item['question'][:40]}{'…' if len(item['question']) > 40 else ''}"
        with st.expander(title, expanded=False):
            st.write(f"**Mode:** {item['mode']}")
            st.write(f"**Question:** {item['question']}")
            st.write(f"**Clean query:** {item['clean_query']}")

            st.write("**Answers:**")
            for name, ans in item["answers"].items():
                st.markdown(f"**{name}**")
                st.write(ans)

            with st.expander("Prompts sent to LLM (debug)", expanded=False):
                for name, prm in item["prompts"].items():
                    st.markdown(f"**{name} prompt**")
                    st.text(prm)

q = st.text_input(
    "Question",
    key="question_input",
    placeholder="Ask a question about DAOD 6000-series (Information Management)...",
)

ask_clicked = st.button("Ask")

if ask_clicked and q.strip():
    user_q = clean_ws(q)

    with st.spinner("Rewriting question…"):
        rq = rewrite_question(user_q)
        clean_query = clean_ws(rq.get("clean_query", user_q))

    if retriever_choice == "all_retrievers":
        run_list = ["keyword_hybrid", "hybrid", "keyword", "entity"]
        mode_label = "all_retrievers"
    else:
        run_list = [retriever_choice]
        mode_label = retriever_choice

    prompts: Dict[str, str] = {}
    answers: Dict[str, str] = {}
    debug_by_retriever: Dict[str, Dict[str, Any]] = {}

    for name in run_list:
        with st.spinner(f"Running pipeline ({name})…"):
            out = _run_one_retriever(name, user_q=user_q, clean_query=clean_query, top_k=int(k))
        prompts[name] = out["prompt"]
        answers[name] = out["answer"]
        debug_by_retriever[name] = out

    # ✅ Answers FIRST (with citations in the text)
    st.subheader("Answer")
    if retriever_choice == "all_retrievers":
        for name in run_list:
            st.markdown(f"### {name}")
            st.write(answers[name])
    else:
        st.write(answers[run_list[0]])

    # ✅ log to sidebar history
    st.session_state.history.append(
        {
            "ts": _now_ts(),
            "question": user_q,
            "clean_query": clean_query,
            "mode": mode_label,
            "prompts": prompts,
            "answers": answers,
        }
    )

    # ✅ DEBUG AFTER answers
    st.divider()
    st.subheader("Debug")

    if retriever_choice != "all_retrievers":
        name = run_list[0]
        dbg = debug_by_retriever.get(name)

        if dbg is not None:
            with st.expander(f"DEBUG ({name}) — Retrieved chunks"):
                st.json(dbg["chunks"])

            with st.expander(f"DEBUG ({name}) — seed vs expanded chunk keys"):
                seed_chunks = dbg["seed_chunks"]
                expanded_chunks = dbg["expanded_chunks"]

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

            with st.expander(f"DEBUG ({name}) — context sent to LLM"):
                st.text(dbg["context"])

    else:
        # All retrievers: show debug per retriever
        for name in run_list:
            dbg = debug_by_retriever.get(name)
            if dbg is None:
                continue

            with st.expander(f"DEBUG ({name}) — Retrieved chunks"):
                st.json(dbg["chunks"])

            with st.expander(f"DEBUG ({name}) — seed vs expanded chunk keys"):
                seed_chunks = dbg["seed_chunks"]
                expanded_chunks = dbg["expanded_chunks"]

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

            with st.expander(f"DEBUG ({name}) — context sent to LLM"):
                st.text(dbg["context"])
