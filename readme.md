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

## Project Goals

- Build a reusable pipeline for ingesting DND policy documents into a graph database.
- Support multiple retrieval modes (keyword, vector, entity-based, hybrid).
- Provide a command-line and web-based interface for asking questions.
- Demonstrate a practical use case of **GraphDB + LLMs** for enterprise/policy knowledge retrieval.

---

## Architecture & Design

### High-Level Architecture

CORAL follows a staged pipeline from source documents to final Q&A:

1. **Indexing / Cataloging**
2. **Scraping**
3. **Parsing / Structuring**
4. **Chunking**
5. **Entity / QA extraction (optional/enhanced)**
6. **Embedding generation**
7. **Neo4j ingestion + retrieval + answer generation**

### Pipeline Stages (1–7)

#### Stage 1 — Indexing / Catalog Building
A catalog/index of target DAOD pages is built so the pipeline knows which documents to ingest.

**Purpose**
- Track source URLs
- Avoid duplicate processing
- Support reproducible ingestion runs

---

#### Stage 2 — Scraping
The scraper downloads policy pages (and optionally related content) from official sources.

**Purpose**
- Collect raw HTML/text content
- Preserve source links and metadata
- Prepare input for parsing

---

#### Stage 3 — Parsing / Document Structuring
Raw content is parsed into a structured format.

Typical output structure:
- **Document**
  - title
  - DAOD number
  - source URL
- **Section**
  - heading/subheading
  - section text

**Purpose**
- Preserve document hierarchy
- Improve retrieval precision vs flat text blobs

---

#### Stage 4 — Chunking
Section text is split into smaller chunks for embedding and retrieval.

**Why chunking matters**
- LLMs and vector search perform better on focused passages
- Reduces irrelevant context in final answers
- Makes ranking more precise

Typical chunk metadata may include:
- `chunk_id`
- `doc_title`
- `section_title`
- `chunk_index`
- `start_char`
- `text`

---

#### Stage 5 — Entity / QA Extraction (Graph Enrichment)
CORAL can enrich the graph by extracting:
- **Entities** (people, organizations, roles, systems, etc.)
- **QAPair** nodes (question-answer style distilled knowledge, if enabled)

**Purpose**
- Support graph-aware retrieval
- Improve semantic matching
- Enable future GraphRAG expansion beyond chunk-only retrieval

---

#### Stage 6 — Embedding Generation
Embeddings are generated for chunks (and optionally QA pairs) using a local embedding model (e.g., Ollama + `nomic-embed-text`).

**Purpose**
- Enable semantic/vector search
- Retrieve relevant content even when wording differs

---

#### Stage 7 — Neo4j Ingestion, Retrieval, and Answering
Structured data is loaded into **Neo4j** with constraints and indexes:
- Uniqueness constraints for key node IDs
- Full-text index for keyword search
- Vector indexes for semantic search

Retrievers then query Neo4j and return context to the answer-generation layer.

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
3. builds a prompt with context,
4. calls the LLM,
5. returns a grounded answer.

---

## Python Files (Core Application Components)

> Below is based on the project versions we worked on together. If any filename differs in your current repo, update this section.

### Main App / Entry Points

- **`app.py`**  
  Streamlit UI for interactive question answering.  
  Lets the user type a question and (in newer versions) choose a retriever mode.

- **`ask.py`**  
  Command-line Q&A interface.  
  Takes a question, retrieves relevant chunks, and generates an answer using the configured LLM.

---

### Retrieval Modules (examples used in your project)

- **`retrieve_keyword.py`**  
  Keyword / full-text retrieval against Neo4j.

- **`retrieve_entity.py`**  
  Entity-aware retrieval using graph nodes/relations.

- **`retrieve_hybrid.py`**  
  Hybrid retrieval (typically vector + keyword fusion).

- **`retrieve_keyword_hybrid.py`**  
  Keyword retrieval + HYbrid retrieval

- **`retrieve.py`** *(if present in your current version)*  
  Wrapper/standalone retrieval entry point (depending on your refactor stage).

---

### Query Expansion / Context Utilities (if present)

- **`question_rewrite.py`**  
  Rewrites user questions to improve retrieval (e.g., clearer keywords/phrasing).

- **`expand_section.py`**  
  Expands seed chunk results to include surrounding section context when useful.

---

### Ingestion / Build Pipeline Scripts (examples, adjust to your repo)

Depending on your current implementation, you may also have scripts for:
- scraping DAOD pages
- parsing/structuring documents
- chunking
- embedding generation
- Neo4j loading/upserts

If available, add them here with a one-line description for maintainability.

---

## What is a Graph DB? (Aside)

A **Graph Database (GraphDB)** stores data as **nodes** and **relationships** rather than only rows and columns.

### Why use a Graph DB for CORAL?

CORAL is not just storing plain text — it stores structured knowledge such as:

- **Documents**
- **Sections**
- **Chunks**
- **Entities**
- **QA Pairs** (optional)
- Relationships between them (e.g., *belongs to*, *mentions*, *derived from*)

This is useful because:

- It preserves document hierarchy (Document → Section → Chunk)
- It supports richer retrieval (entity and relationship-aware search)
- It makes future GraphRAG enhancements easier (traversals, reasoning paths, citation paths)

### Example (Conceptual)

- A **Document** node can have many **Section** nodes  
- Each **Section** can have many **Chunk** nodes  
- A **Chunk** may mention one or more **Entity** nodes  
- A **QAPair** may be linked back to the chunk(s) it came from

### Screenshot of Nodes
> Insert a screenshot from Neo4j Browser / Bloom here showing your node types and relationships.

Suggested caption:
> *Figure X. Neo4j graph view of CORAL nodes (Document, Section, Chunk, Entity, QAPair) and their relationships.*

---

## Data Model (Neo4j)

### Node Types (used / planned)

- `Document`
- `Section`
- `Chunk`
- `Entity`
- `QAPair`
- `TableImage` *(if used in your pipeline)*

### Relationships (current CORAL schema)

CORAL preserves document structure and extracted knowledge using graph relationships:

- `(:Document)-[:HAS_SECTION]->(:Section)`
- `(:Section)-[:CONTAINS]->(:Chunk)`
- `(:Chunk)-[:MENTIONS]->(:Entity)`

---

## Indexes and Constraints (Neo4j)

CORAL uses Neo4j constraints and indexes for both data integrity and retrieval performance.

### Typical Constraints
- Unique IDs for:
  - `Document`
  - `Chunk`
  - `Entity`
  - `QAPair`
  - `TableImage`

### Typical Indexes
- **Full-text index** for keyword retrieval on document/chunk/question/answer fields
- **Vector index** on `Chunk.embedding`
- **Vector index** on `QAPair.embedding` (if enabled)

This enables:
- fast keyword search,
- semantic retrieval,
- and hybrid ranking approaches.

---

## Configuration & Deployment Guide

### Configuration

CORAL uses environment variables (typically from a `.env` file) for services and models.

Example variables (adjust to your current setup):

- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `OLLAMA_HOST`
- `OLLAMA_EMBED_MODEL`
- `LLM_MODEL`
- `TOPK`
- `HTTP_TIMEOUT`
- `OLLAMA_TIMEOUT`
