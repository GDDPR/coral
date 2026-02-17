# ask.py
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import requests

from retrieve_keyword import retrieve_keyword
from retrieve_entity import retrieve_entity
from retrieve_hybrid import retrieve_hybrid
from retrieve_keyword_hybrid import retrieve_keyword_hybrid

# ---- Ollama ----
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:12b")

HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "60"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))

# ---- Defaults ----
TOPK_DEFAULT = int(os.getenv("TOPK", "8"))
MAX_CHARS_PER_CHUNK = int(os.getenv("MAX_CHARS_PER_CHUNK", "2000"))


def clean_ws(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()


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


def try_read_stdin_json() -> Optional[Dict[str, Any]]:
    if sys.stdin.isatty():
        return None
    raw = sys.stdin.read().strip()
    if not raw:
        return None
    return json.loads(raw)


def pick_chunks(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Your retrievers return {"results": [...] }.
    Older pipeline may return {"retrieval": {"chunks": [...]}}.
    Support both (prefer results).
    """
    if not isinstance(data, dict):
        return []
    res = data.get("results")
    if isinstance(res, list):
        return res
    retr = data.get("retrieval") or {}
    chunks = retr.get("chunks")
    if isinstance(chunks, list):
        return chunks
    return []


def build_context(chunks: List[Dict[str, Any]], top_k: int, max_chars_per_chunk: int) -> str:
    blocks: List[str] = []

    for ch in (chunks or [])[: int(top_k)]:
        cid = clean_ws(ch.get("chunk_id") or ch.get("id") or "")
        txt = clean_ws(ch.get("text") or "")

        doc_title = clean_ws(ch.get("doc_title") or "")
        section_title = clean_ws(ch.get("section_title") or ch.get("section") or "")
        daod_number = clean_ws(ch.get("daod_number") or "")

        if not cid or not txt:
            continue

        if len(txt) > int(max_chars_per_chunk):
            txt = txt[: int(max_chars_per_chunk)].rstrip()

        header_parts: List[str] = []
        if doc_title:
            header_parts.append(doc_title)
        elif daod_number:
            header_parts.append(daod_number)

        if section_title:
            header_parts.append(f"Section: {section_title}")

        header = " | ".join(header_parts) if header_parts else "Context"

        blocks.append(f"{header}\n[chunk_id={cid}]\n{txt}\n")

    return "\n".join(blocks).strip()


def build_prompt(question: str, context: str) -> str:
    return f"""SYSTEM:
You are a retrieval-based assistant for the Defence Administrative Orders and Directives (DAOD), and only for the 6000 series focused on Information Management.

You MUST answer only using the provided context.
If the answer is not contained in the context, say:
"I have searched the content of my DAOD 6000 database and found no matching information — I cannot answer."

Do NOT use prior knowledge.
Do NOT speculate.
Cite which chunk_ids and (if shown) sections support your answer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


def main() -> None:
    data = try_read_stdin_json()

    if data is None:
        q = clean_ws(input("Question> "))
        # default retriever (as you wanted)
        data = retrieve_keyword_hybrid(q, top_k=TOPK_DEFAULT)

    question = clean_ws(data.get("question") or data.get("query") or "")
    chunks = pick_chunks(data)

    context = build_context(chunks, top_k=TOPK_DEFAULT, max_chars_per_chunk=MAX_CHARS_PER_CHUNK)
    prompt = build_prompt(question, context)

    answer = ollama_generate(prompt).strip()
    print(answer)


if __name__ == "__main__":
    main()
