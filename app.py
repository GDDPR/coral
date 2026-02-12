# count_unique_chunk_ids.py
import json
from pathlib import Path

JSONL_PATH = Path("data/chunks.jsonl")  # change if needed

def main() -> None:
    total_lines = 0
    nonempty_lines = 0
    missing_id = 0
    ids = set()

    with JSONL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            nonempty_lines += 1

            obj = json.loads(line)
            cid = obj.get("id")
            if not cid:
                missing_id += 1
                continue

            ids.add(cid)

    print(f"File: {JSONL_PATH.resolve()}")
    print(f"Total lines: {total_lines}")
    print(f"Non-empty JSON lines: {nonempty_lines}")
    print(f"Lines missing 'id': {missing_id}")
    print(f"Unique ids: {len(ids)}")

if __name__ == "__main__":
    main()
