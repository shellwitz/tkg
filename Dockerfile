FROM neo4j:5

ARG GDS_JAR_URL=https://graphdatascience.ninja/neo4j-graph-data-science-2.13.7.jar

# Vendor GDS plugin at build time to avoid runtime download corruption.
RUN apt-get update \
  && apt-get install -y curl \
  && rm -rf /var/lib/apt/lists/* \
  && mkdir -p /var/lib/neo4j/plugins \
  && curl -fL --retry 5 --retry-delay 2 --retry-connrefused "$GDS_JAR_URL" \
    -o /var/lib/neo4j/plugins/graph-data-science.jar

# Copy schema and custom entrypoint
COPY schema.cypher /init/schema.cypher
COPY neo4j-entrypoint.sh /neo4j-entrypoint.sh
RUN chmod +x /neo4j-entrypoint.sh

# Use a wrapper that starts Neo4j, waits until ready, applies schema, then stays running
ENTRYPOINT ["/neo4j-entrypoint.sh"]
CMD ["neo4j", "console"]
