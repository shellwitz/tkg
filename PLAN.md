Plan: Minimal Temporal KG RAG MVP
This plan defines a minimal temporal knowledge graph RAG system using Neo4j for both graph storage and vector search. It includes temporal ISO range modeling, entity deduplication logic, OpenAI client usage via .env, and a query pipeline that extracts time/entities, filters by time, retrieves a focused subgraph, and answers with LLM context. It also captures a draft Cypher schema for constraints and indexes.

Steps 1. Define KG schema, constraints, and vector indexes in schema.cypher using the draft below.
Configure model endpoints and thresholds in settings.py and read from .env.
Implement ingestion deduplication flow with normalize → vector search → thresholds → optional LLM gate → alias.
Implement retrieval: time extraction → strict ISO filter → optional BM25 → PPR.
Wire prompts and model calls in prompts.py using OpenAI client.
Further Considerations 1. Keep time tags and edge-embedding-biased PPR as post‑MVP upgrades.
Store relation_text on edges for interpretability and potential lexical match.
Draft Cypher schema for schema.cypher
Constraints and indexes (Neo4j 5.x):

Node labels:
Entity for canonical entities.
Chunk for text chunks with embeddings.
Source for documents (optional minimal node).
Relationship types:
RELATED_TO between entities, with relation_text, start_date, end_date, source_id.
MENTIONS from Chunk → Entity.
FROM_SOURCE from Chunk → Source.
Draft Cypher:

Create constraints:
Entity(entity_id) unique
Chunk(chunk_id) unique
Source(source_id) unique
Create indexes:
Full‑text index on Entity.name + Entity.aliases
Full‑text index on Chunk.text
Range index on RELATED_TO.start_date, RELATED_TO.end_date
Vector index on Chunk.embedding
Skeleton Cypher (outline):

CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE;
CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE;
CREATE CONSTRAINT source_id_unique IF NOT EXISTS FOR (s:Source) REQUIRE s.source_id IS UNIQUE;
CREATE FULLTEXT INDEX entity_name_aliases IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.aliases];
CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text];
CREATE INDEX rel_time_range IF NOT EXISTS FOR ()-[r:RELATED_TO]-() ON (r.start_date, r.end_date);
CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS FOR (c:Chunk) ON (c.embedding) OPTIONS { indexConfig: { vector.dimensions: <D>, vector.similarity_function: 'cosine' } };