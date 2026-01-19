FROM neo4j:5

# Copy schema and custom entrypoint
COPY schema.cypher /init/schema.cypher
COPY neo4j-entrypoint.sh /neo4j-entrypoint.sh
RUN chmod +x /neo4j-entrypoint.sh

# Use a wrapper that starts Neo4j, waits until ready, applies schema, then stays running
ENTRYPOINT ["/neo4j-entrypoint.sh"]
CMD ["neo4j", "console"]
