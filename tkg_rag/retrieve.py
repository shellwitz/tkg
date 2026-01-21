import os
from typing import Dict, Iterable, List, Optional, Tuple

from .ingest import (
    TimestampRange,
    _entity_type_strict_dedup,
    _escape_lucene_query,
    _neo4j_driver,
    embed_texts,
    parse_timestamp_range,
)
from .query_extraction import QueryEntity, extract_query_entities, is_time_entity
from .text_utils import iou, tokens


def _chunk_vector_k() -> int:
    return int(os.getenv("CHUNK_VECTOR_K", "8"))


def _chunk_vector_threshold() -> float:
    return float(os.getenv("CHUNK_VECTOR_THRESHOLD", "0.7"))


def _relation_vector_k() -> int:
    return int(os.getenv("RELATION_VECTOR_K", "12"))


def _relation_vector_threshold() -> float:
    return float(os.getenv("RELATION_VECTOR_THRESHOLD", "0.0"))


def _ppr_damping() -> float:
    return float(os.getenv("PPR_DAMPING", "0.85"))


def _ppr_max_iter() -> int:
    return int(os.getenv("PPR_MAX_ITER", "20"))


def _rrf_k() -> int:
    return int(os.getenv("RRF_K", "60"))


def _entity_bm25_k() -> int:
    return int(os.getenv("ENTITY_BM25_K", "5"))


def _entity_iou_threshold() -> float:
    return float(os.getenv("ENTITY_IOU_THRESHOLD", "0.5"))


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


def _time_overlaps(start_date: Optional[str], end_date: Optional[str], time_range: TimestampRange) -> bool:
    if not time_range.start_date and not time_range.end_date:
        return True
    if time_range.start_date and end_date and end_date < time_range.start_date:
        return False
    if time_range.end_date and start_date and start_date > time_range.end_date:
        return False
    return True


def search_relations(
    tx,
    query_embedding: List[float],
    k: int,
    min_score: float,
) -> List[Dict[str, object]]:
    query = """
    CALL db.index.vector.queryRelationships('relation_embedding', $k, $embedding)
    YIELD relationship, score
    WHERE score >= $min_score
    RETURN id(relationship) AS rel_id,
           score AS similarity,
           relationship.relation_text AS relation_text,
           relationship.start_date AS start_date,
           relationship.end_date AS end_date,
           relationship.source_id AS source_id,
           relationship.chunk_id AS chunk_id,
           id(startNode(relationship)) AS source_node_id,
           id(endNode(relationship)) AS target_node_id,
           startNode(relationship).entity_id AS source_entity_id,
           endNode(relationship).entity_id AS target_entity_id,
           startNode(relationship).name AS source_name,
           endNode(relationship).name AS target_name,
           startNode(relationship).entity_type AS source_type,
           endNode(relationship).entity_type AS target_type
    ORDER BY score DESC
    """
    result = tx.run(query, k=k, embedding=query_embedding, min_score=min_score)
    return [record.data() for record in result]


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


def link_entities_bm25(tx, entities: List[QueryEntity]) -> List[str]:
    entity_ids: List[str] = []
    for entity in entities:
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
            k=_entity_bm25_k(),
        )
        incoming_toks = tokens(entity.name)
        for row in result:
            node = row["node"]
            aliases = node.get("aliases") or []
            iou_alias = 0.0
            for alias in aliases:
                alias_toks = tokens(alias)
                if len(aliases) > 1 and len(alias_toks) == 1:
                    continue
                iou_alias = max(iou_alias, iou(incoming_toks, alias_toks))
            if iou_alias < _entity_iou_threshold():
                continue
            entity_id = node.get("entity_id")
            if entity_id:
                entity_ids.append(entity_id)
    return entity_ids


def edges_for_entities(
    tx,
    entity_ids: Iterable[str],
    time_range: TimestampRange,
) -> List[Dict[str, object]]:
    ids = list(entity_ids)
    if not ids:
        return []
    query = """
    MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
    WHERE (a.entity_id IN $entity_ids OR b.entity_id IN $entity_ids)
      AND ($start IS NULL OR r.end_date IS NULL OR r.end_date >= $start)
      AND ($end IS NULL OR r.start_date IS NULL OR r.start_date <= $end)
    RETURN id(r) AS rel_id,
           0.0 AS similarity,
           r.relation_text AS relation_text,
           r.start_date AS start_date,
           r.end_date AS end_date,
           r.source_id AS source_id,
           r.chunk_id AS chunk_id,
           id(a) AS source_node_id,
           id(b) AS target_node_id,
           a.entity_id AS source_entity_id,
           b.entity_id AS target_entity_id,
           a.name AS source_name,
           b.name AS target_name,
           a.entity_type AS source_type,
           b.entity_type AS target_type
    """
    result = tx.run(
        query,
        entity_ids=ids,
        start=time_range.start_date,
        end=time_range.end_date,
    )
    return [record.data() for record in result]


def fetch_chunks(tx, chunk_ids: List[str]) -> Dict[str, str]:
    if not chunk_ids:
        return {}
    query = """
    MATCH (c:Chunk)
    WHERE c.chunk_id IN $chunk_ids
    RETURN c.chunk_id AS chunk_id, c.text AS text
    """
    result = tx.run(query, chunk_ids=chunk_ids)
    return {record["chunk_id"]: record["text"] for record in result}


def run_ppr_gds(
    tx,
    node_ids: List[int],
    rel_ids: List[int],
    seed_node_ids: List[int],
) -> Dict[str, float]:
    if not node_ids or not rel_ids or not seed_node_ids:
        return {}

    graph_name = "ppr_tmp"
    exists = tx.run("CALL gds.graph.exists($name) YIELD exists", name=graph_name).single()
    if exists and exists["exists"]:
        tx.run("CALL gds.graph.drop($name, false)", name=graph_name)

    node_query = "UNWIND $node_ids AS id RETURN id"
    rel_query = """
    UNWIND $rel_ids AS rel_id
    MATCH (a)-[r]->(b)
    WHERE id(r) = rel_id
    RETURN id(a) AS source, id(b) AS target
    """

    tx.run(
        "CALL gds.graph.project.cypher($name, $node_query, $rel_query, {parameters: {node_ids: $node_ids, rel_ids: $rel_ids}})",
        name=graph_name,
        node_query=node_query,
        rel_query=rel_query,
        node_ids=node_ids,
        rel_ids=rel_ids,
    )

    result = tx.run(
        "CALL gds.pageRank.stream($name, {maxIterations: $max_iter, dampingFactor: $damping, sourceNodes: $seed_nodes}) "
        "YIELD nodeId, score "
        "RETURN gds.util.asNode(nodeId).entity_id AS entity_id, score",
        name=graph_name,
        max_iter=_ppr_max_iter(),
        damping=_ppr_damping(),
        seed_nodes=seed_node_ids,
    )
    scores = {record["entity_id"]: record["score"] for record in result}

    tx.run("CALL gds.graph.drop($name)", name=graph_name)
    return scores


def score_edges(time_valid_relations: List[Dict[str, object]], ppr_scores: Dict[str, float]) -> List[Dict[str, object]]:
    edges: List[Dict[str, object]] = []
    for hit in time_valid_relations:
        source_score = ppr_scores.get(hit["source_entity_id"], 0.0)
        target_score = ppr_scores.get(hit["target_entity_id"], 0.0)
        edge_score = source_score + target_score
        if edge_score <= 0:
            continue
        edge = dict(hit)
        edge["edge_score"] = edge_score
        edges.append(edge)
    return edges


def rrf_fuse(
    edge_ranked: List[Dict[str, object]],
    chunk_ranked: List[Dict[str, object]],
    k: int,
) -> List[Dict[str, object]]:
    scores: Dict[Tuple[str, str], float] = {}
    items: Dict[Tuple[str, str], Dict[str, object]] = {}

    for rank, edge in enumerate(edge_ranked, start=1):
        key = ("edge", str(edge.get("rel_id")))
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        payload = dict(edge)
        payload["kind"] = "edge"
        items[key] = payload

    for rank, chunk in enumerate(chunk_ranked, start=1):
        key = ("chunk", str(chunk.get("chunk_id")))
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        payload = dict(chunk)
        payload["kind"] = "chunk"
        items[key] = payload

    ranked_keys = sorted(scores.keys(), key=lambda k_: scores[k_], reverse=True)
    return [items[k] for k in ranked_keys]


def format_context(items: List[Dict[str, object]]) -> str:
    if not items:
        return "No matching context found."
    lines: List[str] = []
    for item in items:
        if item.get("kind") == "chunk":
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            lines.append(f"[chunk:{item.get('chunk_id')}] {text}")
        else:
            rel_text = str(item.get("relation_text") or "").strip()
            chunk_id = item.get("chunk_id") or "unknown"
            lines.append(
                f"[edge:{item.get('rel_id')}] " # maybe adding the stuff below, but I think it only adds noise
                #f"({item.get('source_name')}, {item.get('source_type')}) -> "
                #f"({item.get('target_name')}, {item.get('target_type')}):\n"
                f"{rel_text}\n"
                f"source: {chunk_id}"
            )
    return "\n".join(lines)

def edge_search(session, query_embedding: List[float], entities: List[QueryEntity], time_range: TimestampRange, max_edges: int) -> List[Dict[str, object]]:
    relation_hits = session.execute_read(
        search_relations,
        query_embedding,
        _relation_vector_k(),
        _relation_vector_threshold(),
    )

    matched_entity_ids = session.execute_read(link_entities_bm25, entities)
    alias_edges = session.execute_read(edges_for_entities, matched_entity_ids, time_range)

    by_rel_id: Dict[int, Dict[str, object]] = {
        hit["rel_id"]: hit for hit in relation_hits
    }
    for edge in alias_edges:
        by_rel_id.setdefault(edge["rel_id"], edge)

    combined_relations = list(by_rel_id.values())
    time_valid_relations = [
        hit
        for hit in combined_relations
        if _time_overlaps(hit.get("start_date"), hit.get("end_date"), time_range)
    ]

    node_ids = set()
    rel_ids = []
    seed_node_ids = set()
    for hit in time_valid_relations:
        rel_ids.append(hit["rel_id"])
        node_ids.add(hit["source_node_id"])
        node_ids.add(hit["target_node_id"])
        seed_node_ids.add(hit["source_node_id"])
        seed_node_ids.add(hit["target_node_id"])

    ppr_scores = session.execute_write(
        run_ppr_gds,
        list(node_ids),
        rel_ids,
        list(seed_node_ids),
    )

    edges = score_edges(time_valid_relations, ppr_scores)
    edges.sort(key=lambda e: e.get("edge_score", 0.0), reverse=True)
    edges = edges[:max_edges]

    return edges

def vector_search(session, query_embedding: List[float], max_chunks: int) -> List[Dict[str, object]]:
    chunk_hits = session.execute_read(
        search_chunks,
        query_embedding,
        _chunk_vector_k(),
        _chunk_vector_threshold(),
    )
    chunk_ids = [hit["chunk_id"] for hit in chunk_hits][:max_chunks]
    chunk_texts = session.execute_read(fetch_chunks, chunk_ids)
    chunks: List[Dict[str, object]] = []
    for hit in chunk_hits[:max_chunks]:
        chunks.append({
            "chunk_id": hit["chunk_id"],
            "text": chunk_texts.get(hit["chunk_id"], hit.get("text", "")),
            "score": hit.get("score", 0.0),
        })

    return chunks

def retrieve(question: str, max_edges: int = 50, max_chunks: int = 12) -> Dict[str, object]:
    entities, time_range = extract_query_entities_and_time(question)
    driver = _neo4j_driver()
    with driver.session() as session:
        query_embedding = embed_texts([question])[0]

        edges = edge_search(session, query_embedding, entities, time_range, max_edges)

        chunks = vector_search(session, query_embedding, max_chunks)

        fused = rrf_fuse(edges, chunks, _rrf_k())
        context = format_context(fused)

    driver.close()
    return {
        "question": question,
        "time_range": time_range,
        "edges": edges,
        "chunks": chunks,
        "context": context,
    }
