import json
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple


# =========================
# CONFIG (change these)
# =========================
IN_DIR = Path("data/docs_json")

CHUNKS_OUT = Path("data/chunks.jsonl")          # Chunk nodes
SECTIONS_OUT = Path("data/sections.jsonl")      # Section nodes
LINKS_OUT = Path("data/chunk_links.jsonl")      # Section->Chunk glue

CHUNK_SIZE = 1000
OVERLAP = 200

# If a section is <= MAX_LAST_CHUNK, it becomes ONE chunk.
# Also, when near the end, we avoid producing a last chunk < CHUNK_SIZE by
# extending the previous chunk to the end (so the last chunk can be bigger).
MAX_LAST_CHUNK = 2000

SPLIT_ON_WHITESPACE_ONLY = True
# =========================


def clean_section_title(title: str) -> str:
    t = (title or "").strip()
    t = re.sub(r"^\s*\d+\s*[\.\)\-:]\s*", "", t)
    return t.strip()


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def make_chunk_id(doc_id: str, section_index: int, chunk_index: int, start_char: int, end_char: int) -> str:
    """
    Stable globally-unique 16-hex Chunk id.
    """
    raw = f"{doc_id}|{section_index}|{chunk_index}|{start_char}|{end_char}"
    return _sha256_text(raw)[:16]


def make_section_id(doc_id: str, section_index: int) -> str:
    """
    Deterministic per-document Section id.
    """
    return f"{doc_id}:s{section_index:03d}"


def _snap_slice_whitespace(text: str, start: int, end: int) -> Tuple[str, int, int]:
    """
    Given a raw [start:end] slice, returns:
      - stripped chunk text
      - corrected start_char/end_char offsets in ORIGINAL text for stripped boundaries
    """
    raw_slice = text[start:end]
    chunk_stripped = raw_slice.strip()

    lstrip_count = len(raw_slice) - len(raw_slice.lstrip())
    rstrip_count = len(raw_slice) - len(raw_slice.rstrip())

    start_char = start + lstrip_count
    end_char = end - rstrip_count

    return chunk_stripped, start_char, end_char


def chunk_text_with_spans(
    text: str,
    chunk_size: int,
    overlap: int,
    max_last_chunk: int,
    split_on_whitespace_only: bool = True,
) -> List[Tuple[str, int, int]]:
    """
    Returns (chunk_text, start_char, end_char) offsets into the ORIGINAL `text` (section text).

    Requirements you asked for:
      - Normal chunks target `chunk_size` (1000)
      - Overlap is `overlap` (100)
      - If section length <= max_last_chunk (2000): make ONE chunk
      - If section length >= chunk_size: do NOT produce a last chunk < chunk_size.
        Instead, extend the previous chunk to the end.
      - Last chunk may be up to `max_last_chunk` (because we stop early and soak remainder)
    """
    results: List[Tuple[str, int, int]] = []
    n = len(text)
    if n == 0:
        return results

    # If the whole section can fit as one "big last chunk", do that.
    if n <= max_last_chunk:
        chunk_stripped, s, e = _snap_slice_whitespace(text, 0, n)
        if chunk_stripped:
            results.append((chunk_stripped, s, e))
        return results

    start = 0

    while start < n:
        end_target = min(start + chunk_size, n)

        # If we reached the end, emit and finish
        if end_target >= n:
            chunk_stripped, s, e = _snap_slice_whitespace(text, start, n)
            if chunk_stripped:
                results.append((chunk_stripped, s, e))
            break

        # Decide end boundary (optionally snap to whitespace)
        if split_on_whitespace_only:
            next_ws = text.find(" ", end_target)
            end = end_target if next_ws == -1 else min(next_ws, n)
        else:
            end = end_target

        # Next chunk start with overlap
        next_start = max(0, end - overlap)
        remaining = n - next_start  # tail length if we start next chunk at next_start

        # If the remaining tail is too small (< chunk_size),
        # do NOT create a tiny last chunk: extend current chunk to the end.
        if remaining < chunk_size:
            extended_stripped, es, ee = _snap_slice_whitespace(text, start, n)
            if extended_stripped:
                results.append((extended_stripped, es, ee))
            break

        # If the remaining tail can be the final chunk and <= max_last_chunk,
        # emit current chunk + final chunk, then stop.
        if chunk_size <= remaining <= max_last_chunk:
            cur_stripped, cs, ce = _snap_slice_whitespace(text, start, end)
            if cur_stripped:
                results.append((cur_stripped, cs, ce))

            final_stripped, fs, fe = _snap_slice_whitespace(text, next_start, n)
            if final_stripped:
                results.append((final_stripped, fs, fe))
            break

        # Otherwise, emit a normal chunk and continue
        cur_stripped, cs, ce = _snap_slice_whitespace(text, start, end)
        if cur_stripped:
            results.append((cur_stripped, cs, ce))

        # Advance start with overlap
        start = max(0, end - overlap)

        # Safety: avoid infinite loops if we somehow can't advance
        if start >= n:
            break

    return results


def safe_read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(out_path: Path, record: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    if not IN_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {IN_DIR.resolve()}")

    json_files = sorted(IN_DIR.glob("*.json"))
    if not json_files:
        print(f"No .json files found in {IN_DIR.resolve()}")
        return

    # Clear old outputs
    for p in (CHUNKS_OUT, SECTIONS_OUT, LINKS_OUT):
        if p.exists():
            p.unlink()

    total_chunks = 0
    total_sections = 0
    total_links = 0
    total_docs = 0

    for jf in json_files:
        try:
            doc = safe_read_json(jf)
        except Exception as e:
            print(f"[SKIP] Could not read {jf.name}: {type(e).__name__}: {e}")
            continue

        doc_id = doc.get("id")
        if not doc_id:
            print(f"[SKIP] {jf.name} missing 'id'")
            continue

        sections = doc.get("sections", [])
        if not isinstance(sections, list):
            print(f"[SKIP] {jf.name} 'sections' is not a list")
            continue

        total_docs += 1

        for section_index, sec in enumerate(sections):
            if not isinstance(sec, dict):
                continue

            raw_title = sec.get("title") or ""
            section_title = clean_section_title(raw_title)

            section_text = sec.get("text") or ""
            if not isinstance(section_text, str):
                section_text = str(section_text)

            section_text = section_text.strip()
            if not section_text:
                continue

            section_id = make_section_id(doc_id, section_index)

            # Write Section record
            append_jsonl(
                SECTIONS_OUT,
                {
                    "id": section_id,
                    "doc_id": doc_id,
                    "section_index": section_index,
                    "title": section_title,
                },
            )
            total_sections += 1

            chunk_spans = chunk_text_with_spans(
                section_text,
                chunk_size=CHUNK_SIZE,
                overlap=OVERLAP,
                max_last_chunk=MAX_LAST_CHUNK,
                split_on_whitespace_only=SPLIT_ON_WHITESPACE_ONLY,
            )

            for chunk_index, (chunk_text, start_char, end_char) in enumerate(chunk_spans):
                chunk_id = make_chunk_id(doc_id, section_index, chunk_index, start_char, end_char)

                chunk_record = {
                    "id": chunk_id,
                    "text": chunk_text,
                    "start_char": start_char,
                    "end_char": end_char,
                    "section": section_title,
                    "embedding": [],  # fill later
                    "token_count": len(chunk_text.split()),
                }
                append_jsonl(CHUNKS_OUT, chunk_record)
                total_chunks += 1

                append_jsonl(
                    LINKS_OUT,
                    {
                        "doc_id": doc_id,
                        "section_id": section_id,
                        "chunk_id": chunk_id,
                    },
                )
                total_links += 1

    print("Done.")
    print(f"  Docs processed:     {total_docs}")
    print(f"  Sections written:   {total_sections}")
    print(f"  Chunks written:     {total_chunks}")
    print(f"  Links written:      {total_links}")
    print(f"  Output chunks:      {CHUNKS_OUT.resolve()}")
    print(f"  Output sections:    {SECTIONS_OUT.resolve()}")
    print(f"  Output links:       {LINKS_OUT.resolve()}")


if __name__ == "__main__":
    main()
