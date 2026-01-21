Plan: Temporal KG RAG with Personalized PageRank (PPR)
This plan rewrites the MVP around the TG-RAG PPR pipeline described in the paper and maps it to this repo’s current design (Neo4j storage + vector search + LLM-based extraction). We will try Neo4j GDS first for PPR, with a Python fallback only if needed.

Overview of the PPR Retrieval Flow (per paper)
1) Query-centric time identification: extract all temporal expressions from the query and derive a time scope Tq.
2) Dynamic subgraph positioning: compute query embedding eq and retrieve top-K relation edges by cosine similarity to edge embeddings; these edges define a query subgraph GKq = (Vq, Eq). Filter edges by time scope Tq and collect their incident entities as seed set Vtq.
   - Hybrid seed expansion: also retrieve top-K chunks by query→chunk embedding and add the mentioned entities into the seed set (lower priority).
3) Local retrieval (PPR): run Personalized PageRank on GKq with personalization vector over Vtq, yielding entity scores s(v). Score edges by time validity: s(e) = 1[time in Tq] * (s(v1) + s(v2)).
4) Chunk scoring: for each chunk c, score it by the edges extracted from c, weighting edges by query-edge similarity gamma_e. Rank chunks and pack until the context budget.

Implementation Plan (project-specific)
Step 0: Data model changes required for PPR
- Add relation embedding storage for edges (relation_text embedding), and link edges to their source chunk.
- Minimal schema changes:
  - Relationship property: r.relation_embedding (vector) OR store embedding on a relation node if Neo4j version limits vector index on relationships.
  - Relationship property: r.chunk_id (the chunk from which the relation was extracted).
  - If relationship vectors are not supported for vector indexes, create a Relation node (label: Relation) with properties:
    - relation_id, relation_text, start_date, end_date, source_id, chunk_id, embedding
    - edges: (a:Entity)-[:RELATED_TO]->(rel:Relation)-[:TARGETS]->(b:Entity)

Step 1: Ingestion updates for PPR
- When extracting relations, embed each relation_text and persist it.
- Store chunk_id on each relation (or Relation node) to enable E(c).
- Ensure that relation embeddings use the same embedding model as chunks (unless a separate model is configured).
- Keep time range on relations as start_date/end_date for temporal filtering.

Step 2: Query-time subgraph positioning
- Compute query embedding eq.
- Retrieve top-K relations by cosine similarity between eq and relation embeddings.
  - If using Relation nodes: vector index on Relation.embedding.
  - If using relationship vectors and indexes: vector index on relationship property (Neo4j 5.20+ supports relationship vector indexes).
- Define GKq as the induced subgraph containing:
  - Eq: those top-K relations
  - Vq: entities incident to Eq
- Time filtering:
  - Extract time scope Tq using current query_extraction.
  - Filter Eq to time-valid edges: start_date/end_date overlap with Tq.
  - Seed set Vtq = entities incident to time-valid edges.

Step 3: Personalized PageRank (PPR)
- Run PPR on GKq with personalization vector over Vtq (GDS first).
  - If GDS is unavailable, implement PPR in Python on the extracted subgraph adjacency (small K).
  - PPR inputs:
    - Nodes: Vq
    - Edges: Eq (treat as undirected or directed; match paper’s choice)
    - Personalization vector: uniform over Vtq
    - Damping factor (alpha): default 0.85
    - Iterations: 20-50 or until convergence
- Compute s(v) for each entity v in Vq.
- Compute s(e) for each relation e=(v1, v2, τ):
  - s(e) = 1[τ in Tq] * (s(v1) + s(v2))

Step 4: Chunk scoring and selection
- For each chunk c, find E(c): relations extracted from chunk c.
- Compute s(c) = sum_{e in E(c)} (1 + gamma_e) * s(e)
  - gamma_e is the cosine similarity between eq and relation_embedding for edge e.
- Rank chunks by s(c) and greedily pack to context budget.

Step 5: Answer generation
- Pass the selected chunks + the structured edge context to the answer model (tkg_rag/answer.py).
- Include time scope in the prompt and require the model to answer within time bounds.

Minimal Engineering Steps to implement
1) Schema updates:
   - Add vector index for relation embeddings (Relation node or relationship property).
   - Add property for chunk_id on relations.
2) Ingest changes:
   - Embed relation_text and store on relation.
   - Store chunk_id on relation.
3) Retrieval changes:
   - Add top-K relation vector search to get Eq.
   - Add top-K chunk vector search and use mentioned entities as additional seeds.
   - Extract Vq and Vtq, run PPR via GDS.
   - Compute edge and chunk scores; select chunks.
4) Add a small PPR utility (pure Python) in tkg_rag/retrieve.py or a new module.
5) Update tests/scripts to run retrieval against the new pipeline.

Notes / assumptions
- This plan implements the PPR logic without requiring Neo4j GDS.
- If GDS becomes available later, the PPR step can be swapped for gds.pageRank.stream with personalization.
- Relationship vector indexes may not be available in older Neo4j builds; if so, use a Relation node to store embeddings and connect it to entities.
