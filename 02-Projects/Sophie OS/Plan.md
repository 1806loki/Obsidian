## 1) MVP Architecture

### Minimal components needed to launch

**Goal:** unified search + “Ask my knowledge” + basic entity graph over a few connectors.

**MVP components**
1. **Connectors (2–3 to start)**
    - Notion, Google Drive/Docs, Obsidian (local vault upload or Git-based sync)

2. **Ingestion service**
    - Pull content + permissions metadata
    - Normalize to a canonical document model

3. **Processing pipeline**
    - Chunking + metadata enrichment
    - Embeddings
    - Lightweight entity extraction (high precision, limited schema)
        
4. **Storage**
    - Document store (canonical docs + chunks)
    - Vector index (semantic search)
    - Graph store (small but real: entities + links)
        
5. **Retrieval API**
    - Hybrid retrieval: vector + keyword + graph expansion

6. **App**    
    - Search UI + “Ask” UI
    - Source linking + citations
        
7. **Ops basics**
    - Auth (OAuth + workspace concept)
    - Basic observability + jobs monitoring
        
### MVP reference architecture (simple)
```
[Connectors] -> [Ingestion API] -> [Queue] -> [Workers: parse/chunk/embed/extract]
                                  |                |         |       |
                                  v                v         v       v
                            [Doc Store]      [Search Index] [Vector] [Graph]
                                  \____________________ Retrieval API ___________________/
                                                       |
                                                    [Web App]
```

### MVP tech stack (with justification)

|Layer|Recommendation|Why this is a good MVP choice|
|---|---|---|
|Backend API|**FastAPI (Python)**|Fast iteration, good for LLM + data pipelines|
|Queue/Jobs|**Celery + Redis** (or **SQS** if AWS-first)|Simple async pipeline, retries, rate limits|
|Doc store|**Postgres** (JSONB)|One system to start: metadata + chunks + ACL refs|
|Vector|**pgvector** (inside Postgres)|Avoid extra infra early; good enough to launch|
|Full-text|**Postgres FTS** (or Meilisearch)|Basic lexical search + filters|
|Graph|**Neo4j** (small instance) OR **Postgres graph tables**|Neo4j makes graph queries and visualization easy early|
|Auth|**Clerk/Auth0** (or Google OAuth + JWT)|Fast secure auth; enterprise later|
|Observability|OpenTelemetry + hosted logs (or CloudWatch)|Debugging ingestion + retrieval is critical|
|Frontend|**Next.js**|Fast UI delivery, good for auth + search UX|

**CTO decision note:** For MVP, “Postgres + pgvector + minimal graph” keeps complexity down while still proving the hybrid thesis.

---

## 2) Production Architecture (scalable, growth-ready)

### What changes in production

- Ingestion becomes **connector platform** (rate limits, delta sync, webhooks)    
- Processing becomes **event-driven** with backfills and reprocessing
- Storage splits by access pattern: **data lake + OLTP + search + vector + graph** 
- Retrieval becomes **multi-stage** (candidate gen + rerank + graph expansion)
- Multi-tenant isolation and permissioning becomes first-class
    

### Production reference architecture

```
                 +--------------------+
                 | Identity / Tenant  |
                 | Auth / Policy      |
                 +---------+----------+
                           |
[Connectors] -> [Connector Orchestrator] -> [Event Bus] -> [Pipeline Workers]
 Notion/Docs     (schedules, webhooks)      (Kafka/SNS)    parse/chunk/embed/NER
  Obsidian                  |                                 |
   Slack                    v                                 v
                    [Raw Content Store]                 [Feature Stores]
                    (S3/GCS + metadata)                 embeddings/entities
                           |                                 |
                           v                                 v
                      [Canonical Store]                +-------------+
                      (Postgres/DocDB)                 | Graph Store |
                           |                           | (Neo4j/...) |
                           v                           +------+------+ 
                    +-------------+                           |
                    | Search      |<--------------------------+
                    | (OpenSearch)|
                    +------+------+ 
                           |
                           v
                    +-------------+
                    | Vector DB   |  (or OpenSearch vector / Pinecone / Weaviate)
                    +------+------+ 
                           |
                           v
                    [Retrieval + RAG API]
                  (hybrid search, rerank, graph expand)
                           |
                           v
                    [Apps + Integrations]
```

### Production data pipelines, storage, retrieval, APIs

**Pipelines**

- **Delta sync**: per source incremental updates (cursor-based pagination, change tokens, webhook triggers)
- **Idempotent processing**: content-hash keys, versioned chunks, deterministic IDs
- **Reprocessing**: re-embed / re-extract with model versioning and “rebuild indices” jobs

**Storage layout**
- **Raw store**: immutable originals, versioned (for audit + reprocessing)
- **Canonical store**: normalized “Document, Block, Chunk, Attachment, EntityMention, Edge”
- **Search store**: lexical + filters (tenant, source, tags, time)
- **Vector store**: semantic retrieval (chunk vectors + entity vectors)
- **Graph store**: entities + edges + provenance

**APIs**
- `/ingest/*` connector operations (sync status, webhook endpoints)
- `/search` hybrid search (filters + vector + FTS)
- `/ask` RAG endpoint (citations + tool routing)
- `/graph/query` entity neighborhood, paths, clustering
- `/insights/*` summaries, digests, anomalies, “what changed”
    

---

## 3) Knowledge Graph Strategy

### Schema approach (practical, evolvable)

Use a **two-layer schema**:

1. **Canonical layer (stable)**
    - `Document`, `Chunk`, `Source`, `User`, `Workspace`, `Tag`, `Task`, `Event`
        
2. **Semantic layer (flexible / typed entities)**
    - `Person`, `Company`, `Project`, `Concept`, `Metric`, `Decision`, `Meeting`, `Claim`
        
**Edges (keep small but meaningful)**

- `MENTIONS (Chunk -> Entity)`
- `DERIVED_FROM (Entity/Edge -> Document/Chunk)` for provenance
- `RELATED_TO (Entity <-> Entity)` (scored, typed when possible)
- `SAME_AS (Entity <-> Entity)` for dedupe
- `AUTHORED_BY`, `SHARED_WITH`, `BELONGS_TO` for access semantics
    
**Critical design principle:** every node/edge stores **provenance** (source doc/chunk, timestamp, extractor version, confidence).
### Entity extraction pipeline (production-safe)

Start high precision, then expand recall.
**Pipeline**
1. **Structure parse**
    - Notion blocks, Docs headings, Markdown structure, chat turns
2. **Mention detection**
    - Hybrid: rules + ML/LLM for candidate mentions
3. **Typing + attributes**
    - Only for a small set of entity types initially
4. **Linking / dedupe**
    - Exact match + alias tables + embedding similarity + graph context
5. **Edge inference**
    - Within-document co-mentions (weak edges) + explicit relations (strong edges)

**Output artifacts**
- `EntityMention`(chunk_id, entity_id, span, type, confidence)
- `Relation`(entity_a, entity_b, relation_type, confidence, provenance)
    
### Embedding integration

You want embeddings at multiple levels:
- **Chunk embeddings**: primary retrieval unit
- **Entity embeddings**: entity-centric retrieval (“find notes about X”)
- **Document embeddings**: fast “broad recall” routing
- **User/profile embeddings** (optional): personalization vectors (see Insight Layer)
### Graph + vector hybrid design (how it actually works)

**Retrieval flow (recommended)**
1. Vector search to get top-k chunks
2. Keyword/filters to ensure precision + recency + source constraints
3. Expand via graph:
    - pull entities from top chunks
    - traverse 1–2 hops to related entities
    - fetch additional chunks mentioning those entities
4. Rerank final candidates (cross-encoder or LLM rerank)
5. Answer with citations + “related nodes” UI
    
This gives:
- vector = fuzzy semantic recall
- graph = structure, explainability, “follow the thread”
- keyword/filters = control, determinism, enterprise needs

---

## 4) Context Preservation Techniques

### Chunking strategy (source-aware)

Chunking must respect structure, not just token size.

**Rules**
- **Docs/Notion**: chunk by heading/section + paragraph groups
- **Markdown/Obsidian**: chunk by heading + block boundaries; preserve backlinks
- **Chat logs**: chunk by conversation windows + speaker turns + topic shifts
- **PDFs**: chunk by page + detected sections (avoid breaking tables)

**Recommended chunk fields**

- `chunk_text`
- `chunk_summary` (optional, generated)
- `section_path` (e.g., H1 > H2 > H3)
- `source_uri` + anchors
- `created_at`, `updated_at`
- `author`, `workspace`, `permissions_ref`
- `content_hash`, `extractor_version`, `embedding_version`
### Metadata enrichment (the multiplier)
Add metadata that improves retrieval quality without needing smarter models:

- Structural: heading path, doc type, block type
- Temporal: created/updated, event time (if meeting notes)
- Social: author, participants, mentioned people
- Project: inferred project tag, status, priority
- Confidence scores: extraction confidence, OCR confidence
### Memory hierarchy (what to store at which granularity)

1. **Raw**: immutable originals
2. **Canonical**: normalized structured representation
3. **Chunks**: retrieval units + embeddings
4. **Summaries**:
    - per-chunk short summary (optional)
    - per-document summary
    - per-entity “entity card” summary
5. **User memory (personalization)**:
    - pinned items
    - recurring topics
    - “important entities” list

---

## 5) Insight Generation Layer

### What insights are “startup valuable” early

Start with insights that feel magical but are feasible:

**Tier 1 (MVP+)**
- “What changed since last week?” (diff summaries across sources)
- Topic digests (auto weekly brief)
- Entity pages (all mentions, timeline, key quotes)
- Duplicate/near-duplicate note detection
- “Unanswered questions” extraction from notes/chats

**Tier 2 (Production)**
- Decision tracker: detect decisions + rationale + owners
- Action item tracker: tasks + due dates + status inferred from updates
- Research synthesis: cluster sources + summarize viewpoints
- Relationship map: people/projects/company graph with timelines
### Analytics and ML opportunities

- **Clustering**: topic modeling over embeddings (per tenant)
- **Trend detection**: rising concepts/entities
- **Anomaly detection**: “sudden spike in incidents/keywords”
- **Quality metrics**: retrieval success, citation coverage, hallucination risk signals
- **Feedback loops**: click-through, save/pin, “this helped” → improves ranking
### Personalization methods (practical)

- **Re-ranking features**: recency, author affinity, source preference, project affinity
- **User embeddings**: represent “what user reads/saves” as a preference vector
- **Explicit controls**: “focus mode” (project X), pinned sources, blocked sources
- **Team-level personalization**: shared pins, team entities, common queries

---

## 6) Visualization Layer

### Tools and UX patterns that win

**Core UX screens**

1. **Universal search**
    - semantic + filters + source scoping
2. **Ask with citations**
    - every claim links to chunks
3. **Entity page**
    - summary, timeline, related entities, top sources
4. **Graph explorer**
    - interactive neighborhood expansion (1–2 hops)
5. **Digest / Insights**
    - weekly changes, key threads, suggested reading

**Visualization tools**
- For graph UI: **Cytoscape.js**, **Sigma.js**, or **React Flow**
- For analytics: simple dashboards first; later add embedded BI if needed

**UX patterns**
- “Explain why I’m seeing this” (vector similarity + graph path + metadata)
- “Narrow scope” chips: source, time, author, project, doc type
- “Follow the thread”: click entity → see all mentions → jump to sources
- “Trust controls”: show extraction confidence + versioning

---

## 7) Tradeoffs & Risks

### Technical risks (and mitigations)

| Risk                          | Why it hurts                     | Mitigation                                                                  |
| ----------------------------- | -------------------------------- | --------------------------------------------------------------------------- |
| Permissions & ACL correctness | Enterprise blocker; trust killer | Centralize permissions model, store `permissions_ref`, filter at query-time |
| Connector rate limits + drift | Sync breaks or becomes stale     | Backoff + incremental tokens + webhook-first where possible                 |
| Entity dedupe errors          | Graph becomes noisy and wrong    | Conservative merge, `SAME_AS` edges, human override, provenance always      |
| Hallucinations in “Ask”       | Trust killer                     | Retrieval-grounded answering, citations required, refusal when low evidence |
| Reprocessing complexity       | Model upgrades break consistency | Version embeddings/extractors; rebuildable indices; immutable raw store     |
### Scaling bottlenecks
- Large tenants: embedding + indexing throughput    
- Graph queries if you do deep traversals
- Cost spikes from frequent re-embedding and LLM extraction
### Cost drivers (what actually burns money)
- Embedding at scale (especially frequent updates)
- LLM-based extraction pipelines (if you do it on every chunk)
- Search infra (OpenSearch) at high QPS + big indexes
- Storage duplication (raw + canonical + chunks + vectors)
    
**Practical cost lever:** extract entities selectively (only high-signal docs) and rely on embeddings for recall; expand graph coverage later.

---

## 8) Competitive Differentiation

To stand out against generic “AI notes/search” tools, differentiate on **trust, structure, and workflows**:

**Defensible angles**
1. **Hybrid explainability**
    - Graph-backed answers: show “why” (paths + sources), not just text generation
2. **Permission-safe enterprise retrieval**
    - Correct ACL enforcement across sources is hard and valuable
3. **Entity-first knowledge**
    - Your UI revolves around entities, timelines, decisions, and tasks—not files
4. **Change intelligence**
    - “What changed?” across Notion/Docs/Slack/Obsidian beats static search
5. **Workflow integrations**
    - Push insights into Slack, email, Jira, GitHub, calendars
6. **User-controlled memory**
    - Explicit pins, project scopes, and “knowledge packs” (curated graphs)

**Startup positioning**
- MVP: “Your cross-app brain with citations.”
- Next: “Decision + research intelligence across your workspace.”
- Enterprise: “Permission-safe knowledge layer with governance and auditability.”

---
### If you want a crisp build plan (execution order)
1. Notion + Google Docs connectors → canonical model
2. Chunk + embed + hybrid search (pgvector + FTS)
3. Ask with citations + feedback logging
4. Add entity mentions + minimal graph (MENTIONS + SAME_AS)
5. Graph expansion in retrieval + entity pages
6. Add “what changed” digests + action/decision extraction (selectively)