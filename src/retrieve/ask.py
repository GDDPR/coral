import json
import os
import re
import sys
import subprocess
from typing import Any, Dict, List, Optional

import requests

# ---- Ollama ----
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:12b")

HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "60"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))

# ---- Context params ----
TOPK = int(os.getenv("TOPK", "8"))
MAX_CHARS_PER_CHUNK = int(os.getenv("MAX_CHARS_PER_CHUNK", "1600"))


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


def call_retrieve_py(question: str) -> Dict[str, Any]:
    # Assumes retrieve.py is in the same directory as ask.py
    here = os.path.dirname(os.path.abspath(__file__))
    retrieve_path = os.path.join(here, "retrieve.py")

    proc = subprocess.run(
        [sys.executable, retrieve_path, question],
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        # show retrieve.py stderr so you can debug it easily
        raise RuntimeError(proc.stderr.strip() or "retrieve.py failed")

    return json.loads(proc.stdout)


def build_context(chunks: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for ch in chunks[:TOPK]:
        cid = clean_ws(ch.get("chunk_id"))
        txt = clean_ws(ch.get("text"))
        if not cid or not txt:
            continue
        if len(txt) > MAX_CHARS_PER_CHUNK:
            txt = txt[:MAX_CHARS_PER_CHUNK].rstrip()
        blocks.append(f"[chunk_id={cid}]\n{txt}\n")
    return "\n".join(blocks).strip()


def build_prompt(question: str, context: str) -> str:
    return f"""SYSTEM: Answer using ONLY the CONTEXT. If not supported, say you don't have enough information.
Cite chunk_ids you used.

QUESTION:
{question}

CONTEXT:
{context}

Return format:
Answer:
...

Citations:
- <chunk_id>
"""


def main() -> None:
    # Mode 2: pipeline JSON input
    data = try_read_stdin_json()

    # Mode 1: interactive -> call retrieve.py
    if data is None:
        q = clean_ws(input("Question> "))
        data = call_retrieve_py(q)

    question = clean_ws(data.get("query"))
    chunks = ((data.get("retrieval") or {}).get("chunks")) or []

    context = build_context(chunks)
    prompt = build_prompt(question, context)

    answer = ollama_generate(prompt).strip()
    print(answer)


if __name__ == "__main__":
    main()
