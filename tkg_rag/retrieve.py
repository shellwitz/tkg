import os
from typing import Dict, Iterable, List, Optional, Tuple

from .ingest import (
    _entity_type_strict_dedup,
    _escape_lucene_query,
    _neo4j_driver,
    embed_texts,
)
from .time_parsing import TimestampRange, parse_timestamp_range
from .query_extraction import QueryEntity, extract_query_entities, is_time_entity
from .text_utils import iou, tokens


def _chunk_vector_k() -> int:
    return int(os.getenv("CHUNK_VECTOR_K", "8"))


def _chunk_vector_threshold() -> float:
    return float(os.getenv("CHUNK_VECTOR_THRESHOLD", "0.7"))


def _relation_vector_k() -> int:
    return int(os.getenv("RELATION_VECTOR_K", "50"))


def _relation_vector_threshold() -> float:
    return float(os.getenv("RELATION_VECTOR_THRESHOLD", "0.0"))


def _ppr_damping() -> float:
    return float(os.getenv("PPR_DAMPING", "0.85"))


def _ppr_max_iter() -> int:
    return int(os.getenv("PPR_MAX_ITER", "100"))

def _ppr_top_k() -> int:
    return int(os.getenv("PPR_TOP_K", "40"))


def _rrf_k() -> int:
    return int(os.getenv("RRF_K", "60"))


def _entity_bm25_k() -> int:
    return int(os.getenv("ENTITY_BM25_K", "5"))


def _entity_iou_threshold() -> float:
    return float(os.getenv("ENTITY_IOU_THRESHOLD", "0.5"))


def _edge_ppr_weight() -> float:
    return float(os.getenv("EDGE_PPR_WEIGHT", "0.6"))


def _edge_similarity_weight() -> float:
    return float(os.getenv("EDGE_SIM_WEIGHT", "0.4"))


def _edge_text_weight() -> float:
    return float(os.getenv("EDGE_TEXT_WEIGHT", "0.2"))


def _merge_time_ranges(ranges: List[TimestampRange]) -> TimestampRange:
    starts = [r.start_date for r in ranges if r.start_date]
    ends = [r.end_date for r in ranges if r.end_date]
    start = min(starts) if starts else None
    end = max(ends) if ends else None
    return TimestampRange(start, end)


def extract_query_entities_and_time(question: str) -> Tuple[List[QueryEntity], TimestampRange]:
    """Extract non-time entities plus a merged time range from the question."""
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
    """Vector search over relation embeddings with a minimum similarity threshold."""
    query = """
    CALL db.index.vector.queryRelationships('relation_embedding', $k, $embedding)
    YIELD relationship, score
    WHERE score >= $min_score
    RETURN id(relationship) AS rel_id,
           score AS similarity,
           relationship.relation_text AS relation_text,
           toString(relationship.start_date) AS start_date,
           toString(relationship.end_date) AS end_date,
           relationship.chunk_ids AS chunk_ids,
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
    """Vector search over chunk embeddings with a minimum similarity threshold."""
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
    """Match query entities to KG entity ids using full-text (BM25) + IoU filtering."""
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
                if len(aliases) > 1 and len(alias_toks) == 1 and alias_toks != incoming_toks:
                    continue
                iou_alias = max(iou_alias, iou(incoming_toks, alias_toks))
            if iou_alias < _entity_iou_threshold():
                continue
            entity_id = node.get("entity_id")
            if entity_id:
                entity_ids.append(entity_id)
    return entity_ids


def fetch_entity_node_ids(tx, entity_ids: List[str]) -> List[int]:
    """Resolve KG entity ids to internal Neo4j node ids."""
    if not entity_ids:
        return []
    query = """
    MATCH (e:Entity)
    WHERE e.entity_id IN $entity_ids
    RETURN id(e) AS node_id
    """
    result = tx.run(query, entity_ids=entity_ids)
    return [record["node_id"] for record in result]


def edges_between_node_ids(
    tx,
    node_ids: Iterable[int],
) -> List[Dict[str, object]]:
    """Fetch all edges between the provided node ids."""
    ids = list(node_ids)
    if not ids:
        return []
    query = """
    MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
    WHERE id(a) IN $node_ids AND id(b) IN $node_ids
    RETURN id(r) AS rel_id,
           0.0 AS similarity,
           r.relation_text AS relation_text,
           toString(r.start_date) AS start_date,
           toString(r.end_date) AS end_date,
           r.chunk_ids AS chunk_ids,
           id(a) AS source_node_id,
           id(b) AS target_node_id,
           a.entity_id AS source_entity_id,
           b.entity_id AS target_entity_id,
           a.name AS source_name,
           b.name AS target_name,
           a.entity_type AS source_type,
           b.entity_type AS target_type
    """
    result = tx.run(query, node_ids=ids)
    return [record.data() for record in result]


def edges_for_entities(
    tx,
    entity_ids: Iterable[str],
    time_range: TimestampRange,
) -> List[Dict[str, object]]:
    """Fetch time-filtered edges incident to the provided entities."""
    ids = list(entity_ids)
    if not ids:
        return []
    query = """
    MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
    WHERE (a.entity_id IN $entity_ids OR b.entity_id IN $entity_ids)
      AND ($start IS NULL OR r.end_date IS NULL OR r.end_date >= date($start))
      AND ($end IS NULL OR r.start_date IS NULL OR r.start_date <= date($end))
    RETURN id(r) AS rel_id,
           0.0 AS similarity,
           r.relation_text AS relation_text,
           toString(r.start_date) AS start_date,
           toString(r.end_date) AS end_date,
           r.chunk_ids AS chunk_ids,
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


def fetch_chunks(tx, chunk_ids: List[str]) -> Dict[str, Dict[str, str]]:
    if not chunk_ids:
        return {}
    query = """
    MATCH (c:Chunk)
    WHERE c.chunk_id IN $chunk_ids
    OPTIONAL MATCH (c)-[:FROM_SOURCE]->(s:Source)
    RETURN c.chunk_id AS chunk_id,
           c.text AS text,
           coalesce(s.name, s.uri, s.source_id) AS source_name
    """
    result = tx.run(query, chunk_ids=chunk_ids)
    return {
        record["chunk_id"]: {
            "text": record["text"],
            "source_name": record.get("source_name"),
        }
        for record in result
    }


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


def run_ppr_gds_fullgraph(
    tx,
    seed_node_ids: List[int],
    top_k: int,
) -> List[Tuple[str, float]]:
    if not seed_node_ids:
        return []

    graph_name = "ppr_tmp"
    exists = tx.run("CALL gds.graph.exists($name) YIELD exists", name=graph_name).single()
    if exists and exists["exists"]:
        tx.run("CALL gds.graph.drop($name, false)", name=graph_name)

    tx.run(
        "CALL gds.graph.project($name, 'Entity', 'RELATED_TO')",
        name=graph_name,
    )

    result = tx.run(
        "CALL gds.pageRank.stream($name, {maxIterations: $max_iter, dampingFactor: $damping, sourceNodes: $seed_nodes}) "
        "YIELD nodeId, score "
        "RETURN gds.util.asNode(nodeId).entity_id AS entity_id, score "
        "ORDER BY score DESC "
        "LIMIT $top_k",
        name=graph_name,
        max_iter=_ppr_max_iter(),
        damping=_ppr_damping(),
        seed_nodes=seed_node_ids,
        top_k=top_k,
    )
    scores = [(record["entity_id"], record["score"]) for record in result]

    tx.run("CALL gds.graph.drop($name)", name=graph_name)
    return scores


def score_edges(
    time_valid_relations: List[Dict[str, object]],
    ppr_scores: Dict[str, float],
    ppr_weight: float,
    similarity_weight: float,
    text_weight: float,
    question_tokens: Optional[set[str]] = None,
) -> List[Dict[str, object]]:
    edges: List[Dict[str, object]] = []
    max_ppr = max(ppr_scores.values()) if ppr_scores else 0.0
    max_sim = 0.0
    for hit in time_valid_relations:
        max_sim = max(max_sim, float(hit.get("similarity", 0.0)))

    total_weight = ppr_weight + similarity_weight + text_weight
    if total_weight > 0:
        ppr_weight = ppr_weight / total_weight
        similarity_weight = similarity_weight / total_weight
        text_weight = text_weight / total_weight
    for hit in time_valid_relations:
        source_score = ppr_scores.get(hit["source_entity_id"], 0.0)
        target_score = ppr_scores.get(hit["target_entity_id"], 0.0)
        ppr_sum = source_score + target_score
        ppr_norm = (ppr_sum / (2 * max_ppr)) if max_ppr > 0 else 0.0
        sim_norm = (float(hit.get("similarity", 0.0)) / max_sim) if max_sim > 0 else 0.0
        text_norm = 0.0
        if question_tokens:
            rel_text = (hit.get("relation_text") or "")
            text_norm = iou(question_tokens, tokens(rel_text))
        edge_score = (ppr_weight * ppr_norm) + (similarity_weight * sim_norm) + (text_weight * text_norm)
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


def format_context(items: List[Dict[str, object]]) -> Tuple[str, dict]:
    chunk_and_edge_map = {}
    inv_chunk_map: Dict[object, int] = {}
    chunk_line_map: Dict[object, int] = {}
    if not items:
        return "No matching context found.", {}
    header = (
        "Context from a temporal knowledge graph. E stands for edge c for chunk with an id[e_id:N] or [c_id:N] "
    )
    lines: List[str] = [header]

    edge_count = 1
    chunk_count = 1
    has_output = False
    for item in items:
        if item["kind"] == "chunk":
            text = item["text"].strip()
            if not text:
                continue

            chunk_id = item["chunk_id"]
            new_chunk_id = inv_chunk_map.get(chunk_id)
            if new_chunk_id is None:
                chunk_and_edge_map[str(chunk_count)] = chunk_id
                new_chunk_id = chunk_count
                inv_chunk_map[chunk_id] = new_chunk_id
                chunk_count += 1

            if not has_output:
                lines.append("")
                has_output = True

            #source_name = (item.get("source_name") or "").strip()
            #source_str = f" | source: {source_name}" if source_name else ""
            lines.append(f"[c_id:{new_chunk_id}] {text}")  # {source_str})
            chunk_line_map[chunk_id] = len(lines) - 1
        else:
            rel_text = item["relation_text"].strip()
            if not rel_text:
                continue

            chunk_ids = item["chunk_ids"]
            chunk_id = chunk_ids[0] if chunk_ids else "unknown"
            new_chunk_id = inv_chunk_map.get(chunk_id)

            if new_chunk_id is None:
                chunk_and_edge_map[str(chunk_count)] = chunk_id
                new_chunk_id = chunk_count
                inv_chunk_map[chunk_id] = new_chunk_id
                chunk_count += 1

            if not has_output:
                lines.append("")
                has_output = True

            edge_id = item["rel_id"]
            chunk_and_edge_map[f"edge{edge_count}"] = edge_id
            start_date = item.get("start_date")
            end_date = item.get("end_date")
            time_str = ""
            if start_date or end_date:
                time_str = f" | time: {start_date or ''} to {end_date or ''}"
            lines.append(
                f"[e_id:{edge_count}] " # maybe adding the stuff below, but I think it only adds noise
                f"{rel_text}"
                # f" | source: {item.get('source_name')} ({item.get('source_type')})"
                # f" | target: {item.get('target_name')} ({item.get('target_type')})"
                # f"{time_str}\n"
                # f"source: {new_chunk_id}"
            )
            source_line_id = chunk_line_map.get(chunk_id)
            if source_line_id is not None:
                lines.append(f"source_id: {source_line_id}")

            edge_count += 1

    return "\n".join(lines), chunk_and_edge_map


def edge_search(
    session,
    query_embedding: List[float],
    entities: List[QueryEntity],
    time_range: TimestampRange,
    max_edges: int,
    question_tokens: Optional[set[str]] = None,
) -> List[Dict[str, object]]:
    relation_hits = session.execute_read(
        search_relations,
        query_embedding,
        _relation_vector_k(),
        _relation_vector_threshold(),
    )

    matched_entity_ids = session.execute_read(link_entities_bm25, entities)
    alias_edges = session.execute_read(edges_for_entities, matched_entity_ids, time_range)
    matched_entity_node_ids = session.execute_read(fetch_entity_node_ids, matched_entity_ids)

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

    seed_node_ids = set(matched_entity_node_ids)
    for hit in time_valid_relations:
        if not seed_node_ids and float(hit.get("similarity", 0.0)) > 0:
            seed_node_ids.add(hit["source_node_id"])
            seed_node_ids.add(hit["target_node_id"])
    if not seed_node_ids:
        for hit in time_valid_relations:
            seed_node_ids.add(hit["source_node_id"])
            seed_node_ids.add(hit["target_node_id"])

    ppr_results = session.execute_write(
        run_ppr_gds_fullgraph,
        list(seed_node_ids),
        _ppr_top_k(),
    )
    ppr_scores = {entity_id: score for entity_id, score in ppr_results}

    edges = score_edges(
        time_valid_relations,
        ppr_scores,
        _edge_ppr_weight(),
        _edge_similarity_weight(),
        _edge_text_weight(),
        question_tokens,
    )
    edges.sort(key=lambda e: e.get("edge_score", 0.0), reverse=True)
    edges = edges[:max_edges]

    return edges

def retrieve_chunks_with_ppr(
    session,
    query_embedding: List[float],
    entities: List[QueryEntity],
    time_range: TimestampRange,
    max_edges: int,
    max_chunks: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Score edges with PPR and aggregate top chunks linked to those edges."""
    relation_hits = session.execute_read(
        search_relations,
        query_embedding,
        _relation_vector_k(),
        _relation_vector_threshold(),
    )

    matched_entity_ids = session.execute_read(link_entities_bm25, entities)
    matched_entity_node_ids = session.execute_read(fetch_entity_node_ids, matched_entity_ids)

    time_seed_node_ids = set(matched_entity_node_ids)
    for hit in relation_hits:
        if _time_overlaps(hit.get("start_date"), hit.get("end_date"), time_range):
            time_seed_node_ids.add(hit["source_node_id"])
            time_seed_node_ids.add(hit["target_node_id"])

    if not time_seed_node_ids:
        return [], []

    ppr_top_k = _ppr_top_k()
    ppr_results = session.execute_write(
        run_ppr_gds_fullgraph,
        list(time_seed_node_ids),
        ppr_top_k,
    )
    ppr_scores = {entity_id: score for entity_id, score in ppr_results}

    ppr_entity_ids = [entity_id for entity_id, _ in ppr_results]
    node_ids = session.execute_read(fetch_entity_node_ids, ppr_entity_ids)
    edges = session.execute_read(edges_between_node_ids, node_ids)

    sim_by_rel_id = {hit["rel_id"]: float(hit.get("similarity", 0.0)) for hit in relation_hits}

    scored_edges: List[Dict[str, object]] = []
    chunk_edge_scores: Dict[str, float] = {}
    chunk_edge_similarities: Dict[str, List[float]] = {}

    for edge in edges:
        if not _time_overlaps(edge.get("start_date"), edge.get("end_date"), time_range):
            continue
        source_score = ppr_scores.get(edge["source_entity_id"], 0.0)
        target_score = ppr_scores.get(edge["target_entity_id"], 0.0)
        edge_score = source_score + target_score
        if edge_score <= 0:
            continue

        edge["edge_score"] = edge_score
        scored_edges.append(edge)

        chunk_ids = edge.get("chunk_ids") or []
        if isinstance(chunk_ids, str):
            chunk_ids = [cid for cid in chunk_ids.split("|") if cid]

        edge_similarity = sim_by_rel_id.get(edge["rel_id"], 0.0)
        for chunk_id in chunk_ids:
            chunk_edge_scores[chunk_id] = chunk_edge_scores.get(chunk_id, 0.0) + edge_score
            chunk_edge_similarities.setdefault(chunk_id, []).append(edge_similarity)

    scored_edges.sort(key=lambda e: e.get("edge_score", 0.0), reverse=True)
    scored_edges = scored_edges[:max_edges]

    chunk_scores: Dict[str, float] = {}
    for chunk_id, edge_score_sum in chunk_edge_scores.items():
        weight = 1.0
        for gamma in chunk_edge_similarities.get(chunk_id, []):
            weight *= (1.0 + gamma)
        chunk_scores[chunk_id] = weight * edge_score_sum

    top_chunk_ids = sorted(chunk_scores.keys(), key=lambda cid: chunk_scores[cid], reverse=True)
    chunk_texts = session.execute_read(fetch_chunks, top_chunk_ids[:max_chunks])
    chunks: List[Dict[str, object]] = []
    for chunk_id in top_chunk_ids[:max_chunks]:
        chunk_payload = chunk_texts.get(chunk_id, {})
        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_payload.get("text", ""),
            "source_name": chunk_payload.get("source_name"),
            "score": chunk_scores.get(chunk_id, 0.0),
        })

    return scored_edges, chunks

def vector_search(session, query_embedding: List[float], max_chunks: int) -> List[Dict[str, object]]:
    """Return top chunks by vector similarity."""
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
        chunk_payload = chunk_texts.get(hit["chunk_id"], {})
        chunks.append({
            "chunk_id": hit["chunk_id"],
            "text": chunk_payload.get("text", hit.get("text", "")),
            "source_name": chunk_payload.get("source_name"),
            "score": hit.get("score", 0.0),
        })

    return chunks

def format_chunk_context(items: List[Dict[str, object]]) -> str:
        """Format chunk payloads into a single prompt-ready context string."""
        chunk_formatter = """---NEW CHUNK ({chunk_id})---
        {chunk_content}
        {chunk_source}
        ---END OF CHUNK---

        """
        processed_chunks = []
        for idx, chunk in enumerate(items, start=1):
            chunk_id = f"chunk{idx}"
            source_name = (chunk.get("source_name") or "").strip()
            source_line = f"source: {source_name}" if source_name else ""
            formatted_chunk = chunk_formatter.format(
                chunk_id=chunk_id,
                chunk_content=chunk["text"],
                chunk_source=source_line,
            )
            processed_chunks.append(formatted_chunk)
        context = "".join(processed_chunks)
        return context

def retrieve(question: str, use_only_vec_search: bool= False) -> Dict[str, object]:
    """Retrieve chunks/edges for a question and return the fused context payload."""
    entities, time_range = extract_query_entities_and_time(question)
    driver = _neo4j_driver()
    with driver.session() as session:
        query_embedding = embed_texts([question])[0]
        if use_only_vec_search:
            max_chunks = 4
            chunks = vector_search(session, query_embedding, max_chunks)
            context = format_chunk_context(chunks)
        else:
            #_, chunks = retrieve_chunks_with_ppr(session, query_embedding, entities, time_range, max_edges, max_chunks) similar method to the one used in the paper
            max_chunks = 1
            chunks = vector_search(session, query_embedding, max_chunks)
            max_edges = 25
            edges = edge_search(session, query_embedding, entities, time_range, max_edges)
            fused = rrf_fuse(edges, 
                            chunks,
                            _rrf_k())
            context, _ = format_context(fused) #the map ids to real ids map isnt used (yet)

    driver.close()
    ret = {
        "question": question,
        "time_range": time_range,
        "chunks": chunks,
        "context": context,
    }
    if not use_only_vec_search:
        ret["edges"] = edges

    return ret
