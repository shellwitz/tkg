// Neo4j 5.x schema for temporal KG + vector search

// Constraints
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
FOR (e:Entity)
REQUIRE e.entity_id IS UNIQUE;

CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
FOR (c:Chunk)
REQUIRE c.chunk_id IS UNIQUE;

CREATE CONSTRAINT source_id_unique IF NOT EXISTS
FOR (s:Source)
REQUIRE s.source_id IS UNIQUE;

// Prime property keys to avoid UnknownPropertyKeyWarning in Community edition.
MERGE (e:Entity {entity_id: "__schema_dummy__"})
SET e.entity_type = "__schema_dummy_type__"
WITH e
DELETE e;

// Full-text indexes
CREATE FULLTEXT INDEX entity_name_aliases IF NOT EXISTS
FOR (e:Entity)
ON EACH [e.name, e.aliases, e.normalized_name]; //maybe too agressive, e.description

CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS
FOR (c:Chunk)
ON EACH [c.text];

// Range index for temporal filtering on relationships
CREATE INDEX rel_time_range IF NOT EXISTS
FOR ()-[r:RELATED_TO]-()
ON (r.start_date, r.end_date);

// Vector index is appended at runtime by neo4j-entrypoint.sh based on EMBEDDING_DIM
