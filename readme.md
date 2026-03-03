# CORAL Documentation

## Overview / Introduction

**CORAL** is a domain-specific **GraphRAG (Graph Retrieval-Augmented Generation)** prototype designed to support question answering over **Canadian Department of National Defence (DND) policy documents**, with a focus on the **DAOD 6000-series** (Information Management / Governance related directives).

The project combines:

- **Document scraping and parsing**
- **Structured chunking (Document → Section → Chunk)**
- **Knowledge graph modeling in Neo4j**
- **Embeddings + vector search**
- **Keyword/full-text retrieval**
- **Hybrid retrieval strategies**
- **LLM-based answer generation**

The goal is to improve policy lookup and understanding by enabling natural-language questions such as:

- “What is ADM(Fin)/CFO responsible for regarding electronic authorizations?”
- “Who approves the implementation and use of SES?”
- “What controls are required for financial transactions?”

Instead of relying on simple keyword search alone, CORAL uses a graph-aware pipeline and multiple retrieval strategies to return relevant context before generating a response.

---

## How CORAL works (end-to-end)

### 1) One-time database build (ingestion pipeline)
This is the “setup” phase. You run it when Neo4j is empty or when you want to rebuild the database.

1. **Scrape DAOD links** and write them to an XML catalog → `catalog.xml`
2. **Parse the catalog**, keep only **non-cancelled** DAODs, fetch the pages, and **split into Sections**
   - Output: per-document JSON files in `docs_json/`
3. **Chunk** the section text into smaller passages
   - Output: `chunks.jsonl` (backup)
4. **Load** Document/Section/Chunk into **Neo4j**
5. **Generate & store embeddings** for chunks using **Ollama embeddings**
6. **Generate & load entities** using **Ollama LLM**

✅ At the end of this stage, **the Neo4j graph is complete** (documents + sections + chunks + embeddings + entities).

### 2) Asking questions (runtime Q&A flow)
This is what happens every time a user asks a question in Streamlit (or via CLI).

1. User enters a **question**
2. User selects a **retrieval mode** (keyword / entity / hybrid / keyword-hybrid)
3. (Optional) the system rewrites/cleans the question (if enabled)
4. The retriever queries Neo4j to find the **top-K most relevant chunks**
5. (Optional) expand chunk hits to **full section context**
6. Build the LLM input:
   - assemble the retrieved context
   - attach **sources/citations** (DAOD + section + link + chunk id, etc.)
7. Call **Ollama LLM** to generate the final answer
8. Display:
   - the answer
   - the sources/citations used

---

## How the graph works (Neo4j)

### Node types
CORAL stores policy content and extracted knowledge as nodes:

- **`Document`**: one DAOD document (DAOD number, title, URL, metadata)
- **`Section`**: a section inside a document (heading + text)
- **`Chunk`**: a small passage of section text used for retrieval + embeddings
- **`Entity`**: extracted “things” mentioned in text (roles, orgs, systems, etc.)
- *(Optional)* **`QAPair`**: distilled question/answer nodes (only if your version enables it)

### Relationships
CORAL keeps the original hierarchy and links extracted knowledge:

- `(:Document)-[:HAS_SECTION]->(:Section)`
- `(:Section)-[:CONTAINS]->(:Chunk)`
- `(:Chunk)-[:MENTIONS]->(:Entity)`

### Why this helps retrieval
- **Hierarchy** lets you expand from a good chunk to the whole section for context.
- **Entities** enable entity-aware retrieval and graph-based filtering.
- **Embeddings on Chunk nodes** enable semantic search (vector retrieval).

---

## Project Goals

- Build a reusable pipeline for ingesting DND policy documents into a graph database.
- Support multiple retrieval modes (keyword, vector, entity-based, hybrid).
- Provide a command-line and web-based interface for asking questions.
- Demonstrate a practical use case of **GraphDB + LLMs** for enterprise/policy knowledge retrieval.

---

## Retrieval & Q&A Design

CORAL supports multiple retrieval strategies (depending on which scripts are enabled in your local version):

- **Keyword retrieval**
- **Entity-based retrieval**
- **Hybrid retrieval** (vector + keyword)
- **Keyword-hybrid retrieval** (combined ranking strategies)

The answer layer then:
1. receives the user question,
2. retrieves top relevant chunks,
3. builds a prompt with context + sources,
4. calls the LLM,
5. returns a grounded answer.

---

## Python Files (Core Application Components)

> If any filename differs in your current repo, update this section.

### Main App / Entry Points

- **`app.py`**  
  Streamlit UI for interactive question answering.

- **`ask.py`**  
  Command-line Q&A interface (optional).

### Retrieval Modules
- `retrieve_keyword.py`
- `retrieve_entity.py`
- `retrieve_hybrid.py`
- `retrieve_keyword_hybrid.py`

### Optional utilities
- `question_rewrite.py`
- `expand_section.py`

---

## Configuration & Deployment Guide

### Configuration
CORAL uses environment variables (usually from a `.env` file) for services and model settings.

Example variables:
- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `OLLAMA_HOST`
- `OLLAMA_EMBED_MODEL`
- `LLM_MODEL`
- `TOPK`
- `MAX_CHARS_PER_CHUNK`
- `VEC_CANDIDATES`
- `HTTP_TIMEOUT`
- `OLLAMA_TIMEOUT`

---

## Running CORAL (Docker Compose)

### Prerequisites
- Docker + Docker Compose installed
- Ports available:
  - `8501` (Streamlit)
  - `7474` + `7687` (Neo4j)
  - `11434` (Ollama)

### 1) Create a `.env` file
Create **`.env`** in the repo root:

```bash
# Neo4j
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=YOUR_PASSWORD

# Ollama
OLLAMA_HOST=http://ollama:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
LLM_MODEL=gemma3:12b

# Retrieval + prompt sizing
TOPK=3
MAX_CHARS_PER_CHUNK=800
VEC_CANDIDATES=20

# Timeouts
HTTP_TIMEOUT=180
OLLAMA_TIMEOUT=180