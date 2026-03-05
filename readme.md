## Configuration & Deployment (Docker Compose)

### Configuration

CORAL is configured using environment variables (typically stored in a `.env` file at the repo root).

**Main variables:**
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` (Neo4j connection)
- `OLLAMA_HOST`, `OLLAMA_EMBED_MODEL`, `LLM_MODEL` (LLM + embeddings via Ollama)
- `TOPK`, `MAX_CHARS_PER_CHUNK`, `VEC_CANDIDATES` (retrieval + context sizing)
- `HTTP_TIMEOUT`, `OLLAMA_TIMEOUT` (timeouts in seconds)

**What the tuning variables mean:**
- `TOPK`: number of chunks retrieved per question (higher = more context, slower)
- `MAX_CHARS_PER_CHUNK`: max characters taken from each chunk when building the prompt
- `VEC_CANDIDATES`: number of vector-search candidates considered before ranking/filtering
- `HTTP_TIMEOUT`: request timeout for network calls (Neo4j / HTTP)
- `OLLAMA_TIMEOUT`: max time allowed for LLM generation

---

### Prerequisites

- Docker + Docker Compose installed
- Available ports:
  - `8501` (Streamlit)
  - `7474` + `7687` (Neo4j Browser + Bolt)
  - `11434` (Ollama)

---

### 1) Create a `.env` file

Create **`.env`** in the repo root (same folder as `docker-compose.yml`):

```bash
# Neo4j
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=YOUR_PASSWORD # change this with password

# Ollama
OLLAMA_HOST=http://ollama:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
LLM_MODEL=gemma3:12b

# Retrieval + prompt sizing
TOPK=3
MAX_CHARS_PER_CHUNK=800
VEC_CANDIDATES=20

# Timeouts (seconds)
HTTP_TIMEOUT=180
OLLAMA_TIMEOUT=180