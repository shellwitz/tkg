import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ingest import (
    TimestampRange,
    _escape_lucene_query,
    _entity_bm25_threshold,
    _entity_type_strict_dedup,
    _entity_vector_k,
    _entity_vector_threshold,
    _neo4j_driver,
    embed_texts,
    embed_entity_text,
    parse_timestamp_range,
)
from query_extraction import QueryEntity, extract_query_entities, is_time_entity


@dataclass
class LinkedEntity:
    entity_id: str
    name: str
    entity_type: str
    score: float
    method: str


def _retrieval_bm25_threshold() -> float:
    return float(os.getenv("RETRIEVAL_ENTITY_BM25_THRESHOLD", _entity_bm25_threshold()))


def _retrieval_vector_threshold() -> float:
    return float(os.getenv("RETRIEVAL_ENTITY_VECTOR_THRESHOLD", _entity_vector_threshold()))


def _chunk_vector_k() -> int:
    return int(os.getenv("CHUNK_VECTOR_K", "8"))


def _chunk_vector_threshold() -> float:
    return float(os.getenv("CHUNK_VECTOR_THRESHOLD", "0.0"))


def _merge_time_ranges(ranges: List[TimestampRange]) -> TimestampRange:
    starts = [r.start_date for r in ranges if r.start_date]
    ends = [r.end_date for r in ranges if r.end_date]
    start = min(starts) if starts else None
    end = max(ends) if ends else None
    return TimestampRange(start, end)


def extract_query_entities_and_time(question: str) -> Tuple[List[QueryEntity], TimestampRange]:
    entities = extract_query_entities(question)
    time_ranges = [parse_timestamp_range(e.name) for e in entities if is_time_entity(e.entity_type)]
    time_range = _merge_time_ranges([r for r in time_ranges if r.start_date or r.end_date])
    non_time_entities = [e for e in entities if not is_time_entity(e.entity_type)]
    return non_time_entities, time_range


def _link_entity_bm25(tx, entity: QueryEntity, k: int = 1) -> Optional[LinkedEntity]:
    query = """
    CALL db.index.fulltext.queryNodes('entity_name_aliases', $query_text)
    YIELD node, score
    WHERE ($type_strict = false OR node.entity_type = $entity_type)
    RETURN node, score
    ORDER BY score DESC
    LIMIT $k
    """
    result = tx.run(
        query,
        query_text=_escape_lucene_query(entity.name),
        entity_type=entity.entity_type,
        type_strict=_entity_type_strict_dedup(),
        k=k,
    )
    row = result.single()
    if not row:
        return None
    if row["score"] < _retrieval_bm25_threshold():
        return None
    node = row["node"]
    return LinkedEntity(
        entity_id=node["entity_id"],
        name=node.get("name") or entity.name,
        entity_type=node.get("entity_type") or entity.entity_type,
        score=row["score"],
        method="bm25",
    )


def _link_entity_vector(tx, entity: QueryEntity) -> Optional[LinkedEntity]:
    embedding = embed_entity_text(entity.name, "")
    query = """
    CALL db.index.vector.queryNodes('entity_embedding', $k, $embedding)
    YIELD node, score
    WHERE ($type_strict = false OR node.entity_type = $entity_type)
    RETURN node, score
    ORDER BY score DESC
    LIMIT 1
    """
    result = tx.run(
        query,
        k=_entity_vector_k(),
        embedding=embedding,
        entity_type=entity.entity_type,
        type_strict=_entity_type_strict_dedup(),
    )
    row = result.single()
    if not row:
        return None
    if row["score"] < _retrieval_vector_threshold():
        return None
    node = row["node"]
    return LinkedEntity(
        entity_id=node["entity_id"],
        name=node.get("name") or entity.name,
        entity_type=node.get("entity_type") or entity.entity_type,
        score=row["score"],
        method="vector",
    )


def link_entities(tx, entities: List[QueryEntity]) -> List[LinkedEntity]:
    linked: Dict[str, LinkedEntity] = {}
    for entity in entities:
        match = _link_entity_bm25(tx, entity)
        if not match:
            match = _link_entity_vector(tx, entity)
        if not match:
            continue
        current = linked.get(match.entity_id)
        if not current or match.score > current.score:
            linked[match.entity_id] = match
    return list(linked.values())


def search_chunks(
    tx,
    query_embedding: List[float],
    k: int,
    min_score: float,
) -> List[Dict[str, object]]:
    query = """
    CALL db.index.vector.queryNodes('chunk_embedding', $k, $embedding)
    YIELD node, score
    WHERE score >= $min_score
    RETURN node.chunk_id AS chunk_id, node.text AS text, score
    ORDER BY score DESC
    """
    result = tx.run(query, k=k, embedding=query_embedding, min_score=min_score)
    return [record.data() for record in result]


def entities_from_chunks(tx, chunk_ids: List[str]) -> List[str]:
    if not chunk_ids:
        return []
    query = """
    MATCH (c:Chunk)
    WHERE c.chunk_id IN $chunk_ids
    MATCH (c)-[:MENTIONS]->(e:Entity)
    RETURN DISTINCT e.entity_id AS entity_id
    """
    result = tx.run(query, chunk_ids=chunk_ids)
    return [record["entity_id"] for record in result]


def fetch_edges(
    tx,
    entity_ids: List[str],
    time_range: TimestampRange,
    limit: int,
) -> List[Dict[str, Optional[str]]]:
    if not entity_ids:
        return []
    query = """
    MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
    WHERE a.entity_id IN $entity_ids
      AND ($start IS NULL OR r.end_date IS NULL OR r.end_date >= $start)
      AND ($end IS NULL OR r.start_date IS NULL OR r.start_date <= $end)
    RETURN a.name AS source,
           a.entity_type AS source_type,
           b.name AS target,
           b.entity_type AS target_type,
           r.relation_text AS relation_text,
           r.start_date AS start_date,
           r.end_date AS end_date,
           r.source_id AS source_id
    LIMIT $limit
    """
    result = tx.run(
        query,
        entity_ids=entity_ids,
        start=time_range.start_date,
        end=time_range.end_date,
        limit=limit,
    )
    return [record.data() for record in result]


def format_edges_as_context(edges: List[Dict[str, Optional[str]]]) -> str:
    if not edges:
        return "No matching relations found."
    lines = []
    for edge in edges:
        start = edge.get("start_date") or "unknown"
        end = edge.get("end_date") or "unknown"
        lines.append(
            f"- {edge.get('source')} ({edge.get('source_type')}) "
            f"-[{edge.get('relation_text')}]→ "
            f"{edge.get('target')} ({edge.get('target_type')}) "
            f"[{start} to {end}] (source: {edge.get('source_id')})"
        )
    return "\n".join(lines)


def retrieve(question: str, max_edges: int = 50) -> Dict[str, object]:
    entities, time_range = extract_query_entities_and_time(question)
    driver = _neo4j_driver()
    with driver.session() as session:
        query_embedding = embed_texts([question])[0]
        chunk_hits = session.execute_read(
            search_chunks,
            query_embedding,
            _chunk_vector_k(),
            _chunk_vector_threshold(),
        )
        chunk_ids = [hit["chunk_id"] for hit in chunk_hits]
        entity_ids = session.execute_read(entities_from_chunks, chunk_ids)

        linked_entities: List[LinkedEntity] = []
        if not entity_ids:
            linked_entities = session.execute_read(link_entities, entities)
            entity_ids = [e.entity_id for e in linked_entities]

        edges = session.execute_read(fetch_edges, entity_ids, time_range, max_edges)
    driver.close()
    return {
        "question": question,
        "time_range": time_range,
        "chunk_hits": chunk_hits,
        "linked_entities": linked_entities,
        "edges": edges,
        "context": format_edges_as_context(edges),
    }
