#!/usr/bin/env bash
set -euo pipefail

# Start Neo4j using the official entrypoint to preserve all behavior
/startup/docker-entrypoint.sh "$@" &
NEO4J_PID=$!

forward() {
  kill -TERM "$NEO4J_PID"
}

trap forward SIGINT SIGTERM

echo "Waiting for Neo4j to be ready on internal port 7687..."
NEO4J_AUTH_PAIR="${NEO4J_AUTH:-neo4j/passworty}"
NEO4J_AUTH_USER="${TKG_NEO4J_USER:-${NEO4J_AUTH_PAIR%%/*}}"
NEO4J_AUTH_PASS="${TKG_NEO4J_PASSWORD:-${NEO4J_AUTH_PAIR#*/}}"

until cypher-shell -a "bolt://localhost:7687" \
  -u "${NEO4J_AUTH_USER}" \
  -p "${NEO4J_AUTH_PASS}" \
  "RETURN 1" >/dev/null 2>&1; do
  sleep 2
done

EMBEDDING_DIM="${EMBEDDING_DIM:-3072}"
TEMP_SCHEMA="/tmp/schema.cypher"

echo "Preparing schema with vector indexes (EMBEDDING_DIM=${EMBEDDING_DIM})..."
cp /init/schema.cypher "$TEMP_SCHEMA"
cat >>"$TEMP_SCHEMA" <<EOF

// Vector index for chunk embeddings (generated at runtime)
CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
FOR (c:Chunk)
ON (c.embedding)
OPTIONS {
  indexConfig: {
    \`vector.dimensions\`: ${EMBEDDING_DIM},
    \`vector.similarity_function\`: 'cosine'
  }
};

// Vector index for relation embeddings (generated at runtime)
CREATE VECTOR INDEX relation_embedding IF NOT EXISTS
FOR ()-[r:RELATED_TO]-()
ON (r.relation_embedding)
OPTIONS {
  indexConfig: {
    \`vector.dimensions\`: ${EMBEDDING_DIM},
    \`vector.similarity_function\`: 'cosine'
  }
};
EOF

echo "Applying schema from ${TEMP_SCHEMA} ..."
cypher-shell -a "bolt://localhost:7687" \
  -u "${NEO4J_AUTH_USER}" \
  -p "${NEO4J_AUTH_PASS}" \
  -f "$TEMP_SCHEMA"
echo "Schema applied."

wait "$NEO4J_PID"
