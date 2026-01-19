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
until cypher-shell -a "bolt://localhost:7687" \
  -u "${NEO4J_USER:-neo4j}" \
  -p "${NEO4J_PASSWORD:-graphiti}" \
  "RETURN 1" >/dev/null 2>&1; do
  sleep 2
done

echo "Applying schema from /init/schema.cypher ..."
cypher-shell -a "bolt://localhost:7687" \
  -u "${NEO4J_USER:-neo4j}" \
  -p "${NEO4J_PASSWORD:-graphiti}" \
  -f /init/schema.cypher
echo "Schema applied."

wait "$NEO4J_PID"
