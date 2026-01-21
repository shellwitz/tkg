import json
import os
import re
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from neo4j import GraphDatabase

from . import prompts
from .llm_client import openai_client
from .settings import EMBEDDING_DIM, EMBEDDING_MODEL, ENTITY_TYPES, LLM_MODEL
from .text_utils import iou, tokens


@dataclass
class ExtractedEntity:
    name: str
    entity_type: str
    description: str


@dataclass
class ExtractedRelation:
    timestamp_entity: str
    source_entity: str
    target_entity: str
    description: str


@dataclass
class TimestampRange:
    start_date: Optional[str]
    end_date: Optional[str]


DEFAULT_TIME_TYPES = ["date", "date_range", "quarter", "year"]
DEFAULT_TIMESTAMP_FORMAT = "ISO-8601 or ISO-like (YYYY, YYYY-MM-DD, YYYY-Qn)"


def chunk_text(text: str, max_chars: int = 1600, overlap: int = 200) -> List[str]:
    if not text:
        return []
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            split_at = text.rfind("\n", start, end)
            if split_at == -1:
                split_at = text.rfind(". ", start, end)
            if split_at > start + 200:
                end = split_at + 1
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        next_start = max(0, end - overlap)
        # Guard against no forward progress (can otherwise loop forever).
        if next_start <= start:
            break
        start = next_start
    return [c for c in chunks if c]


def _build_extraction_prompts() -> Tuple[str, str, Dict[str, str]]:
    tuple_delimiter = "|"
    record_delimiter = ";;"
    system_prompt = prompts.TEMPORAL_ENTITY_EXTRACTION_SYS_PROMPT.format(
        entity_types=", ".join(ENTITY_TYPES),
        tuple_delimiter=tuple_delimiter,
        record_delimiter=record_delimiter,
        timestamp_types=", ".join(DEFAULT_TIME_TYPES),
        timestamp_format=DEFAULT_TIMESTAMP_FORMAT,
    )
    return system_prompt, prompts.TEMPORAL_ENTITY_EXTRACTION_FOLLOWUP_PROMPT, {
        "tuple_delimiter": tuple_delimiter,
        "record_delimiter": record_delimiter,
    }


def extract_entities_and_relations(text: str) -> Tuple[List[ExtractedEntity], List[ExtractedRelation]]:
    system_prompt, user_prompt_template, delimiters = _build_extraction_prompts()
    client = openai_client()
    if not LLM_MODEL:
        raise RuntimeError("LLM_MODEL is not set.")
    user_prompt = user_prompt_template.format(entity_types=", ".join(ENTITY_TYPES), input_text=text)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content or ""
    return parse_extraction_output(raw, delimiters["tuple_delimiter"], delimiters["record_delimiter"])


def parse_extraction_output(
    raw: str, tuple_delimiter: str, record_delimiter: str
) -> Tuple[List[ExtractedEntity], List[ExtractedRelation]]:
    entities: List[ExtractedEntity] = []
    relations: List[ExtractedRelation] = []
    for record in raw.split(record_delimiter):
        record = record.strip().strip(",")
        if not record:
            continue
        record = record.strip("() ")
        parts = [p.strip().strip('"') for p in record.split(tuple_delimiter)]
        if not parts:
            continue
        if parts[0] == "entity":
            if len(parts) >= 4:
                entities.append(ExtractedEntity(parts[1], parts[2], parts[3]))
            elif len(parts) == 3:
                entities.append(ExtractedEntity(parts[1], parts[2], ""))
        elif parts[0] in {"relationship", "event"}:
            if len(parts) >= 5:
                relations.append(ExtractedRelation(parts[1], parts[2], parts[3], parts[4]))
    return entities, relations


def parse_timestamp_range(name: str) -> TimestampRange:
    name = name.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", name):
        return TimestampRange(name, name)
    if re.match(r"^\d{4}$", name):
        return TimestampRange(f"{name}-01-01", f"{name}-12-31")
    q_match = re.match(r"^(?:Q([1-4])\s*(\d{4})|(\d{4})-Q([1-4]))$", name)
    if q_match:
        q = int(q_match.group(1) or q_match.group(4))
        y = int(q_match.group(2) or q_match.group(3))
        start_month = 3 * (q - 1) + 1
        end_month = start_month + 2
        return TimestampRange(f"{y}-{start_month:02d}-01", f"{y}-{end_month:02d}-31")
    return TimestampRange(None, None)


def _neo4j_driver():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7688")
    user = os.getenv("TKG_NEO4J_USER", "neo4j")
    password = os.getenv("TKG_NEO4J_PASSWORD", "passworty")
    return GraphDatabase.driver(uri, auth=(user, password))


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip()).lower()

def _is_time_entity(entity_type: str) -> bool:
    return entity_type == "timestamp" or entity_type in DEFAULT_TIME_TYPES


def _escape_lucene_query(text: str) -> str:
    # Escape Lucene special chars to avoid query parser errors.
    return re.sub(r'([+\-!(){}[\]^"~*?:\\/]|&&|\|\|)', r"\\\1", text)


def _entity_type_strict_dedup() -> bool:
    return os.getenv("ENTITY_DEDUP_TYPE_STRICT", "true").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def search_entity_by_bm25_and_iou(tx, entity) -> Tuple[Optional[dict], float]:
    K = 10
    query = """
    CALL db.index.fulltext.queryNodes('entity_name_aliases', $query_text)
    YIELD node, score
    WHERE ($type_strict = false OR node.entity_type = $entity_type)
    RETURN node, score
    ORDER BY score DESC
    LIMIT $k
    """
    rows = list(tx.run(
        query,
        query_text=_escape_lucene_query(entity.name),
        entity_type=entity.entity_type,
        type_strict=_entity_type_strict_dedup(),
        k=K,
    ))

    incoming_toks = tokens(entity.name)

    best = None
    best_iou = 0.0

    for r in rows:
        node = r["node"]

        aliases = node.get("aliases") or []
        iou_alias = 0.0
        for a in aliases:
            alias_toks = tokens(a)
            if len(aliases) > 1 and len(alias_toks) ==1:
                continue  # skip single-token aliases if multiple aliases exist.
                # EOG Resources Inc -> EOG Resources -> EOG shouldnt further match -> EOG Burgers or so
            iou_alias = max(iou_alias, iou(incoming_toks, alias_toks))


        if iou_alias > best_iou:
            best_iou = iou_alias
            best = node

    return best, best_iou

def upsert_entity(tx, entity) -> str:
    best, best_iou = search_entity_by_bm25_and_iou(tx, entity)

    IOU_THRESHOLD = 0.5  # tune: 0.5–0.8 typical for entity names

    if best and best_iou >= IOU_THRESHOLD:
        entity_id = best["entity_id"]
        aliases = set(best.get("aliases") or [])

        # update aliases
        if entity.name != best.get("name") and entity.name not in aliases:
            aliases.add(entity.name)
            tx.run(
                "MATCH (e:Entity {entity_id: $id}) SET e.aliases = $aliases",
                id=entity_id,
                aliases=sorted(aliases),
            )

        print(f"Merged by IoU={best_iou:.2f}: '{entity.name}' -> '{best.get('name')}'")
        return entity_id

    # 3) create new if no good lexical overlap
    entity_id = str(uuid.uuid4())
    tx.run(
        """
        CREATE (e:Entity {
          entity_id: $entity_id,
          name: $name,
          entity_type: $entity_type,
          description: $description,
          normalized_name: $normalized_name,
          aliases: $aliases
        })
        """,
        entity_id=entity_id,
        name=entity.name,
        entity_type=entity.entity_type,
        description=entity.description,
        normalized_name=_normalize_name(entity.name),
        aliases=[entity.name],
    )
    return entity_id


def create_chunk(tx, text: str, embedding: List[float], source_id: Optional[str]) -> str:
    chunk_id = str(uuid.uuid4())
    tx.run(
        """
        CREATE (c:Chunk {chunk_id: $chunk_id, text: $text, embedding: $embedding})
        """,
        chunk_id=chunk_id,
        text=text,
        embedding=embedding,
    )
    if source_id:
        tx.run(
            """
            MERGE (s:Source {source_id: $source_id})
            WITH s
            MATCH (c:Chunk {chunk_id: $chunk_id})
            MERGE (c)-[:FROM_SOURCE]->(s)
            """,
            source_id=source_id,
            chunk_id=chunk_id,
        )
    return chunk_id


def link_chunk_mentions(tx, chunk_id: str, entity_ids: Iterable[str]) -> None:
    for entity_id in entity_ids:
        tx.run(
            """
            MATCH (c:Chunk {chunk_id: $chunk_id})
            MATCH (e:Entity {entity_id: $entity_id})
            MERGE (c)-[:MENTIONS]->(e)
            """,
            chunk_id=chunk_id,
            entity_id=entity_id,
        )


def create_relationship(
    tx,
    source_id: str,
    source_entity_id: str,
    target_entity_id: str,
    relation_text: str,
    relation_embedding: List[float],
    chunk_id: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> None:
    tx.run(
        """
        MATCH (s:Entity {entity_id: $source_entity_id})
        MATCH (t:Entity {entity_id: $target_entity_id})
        MERGE (s)-[r:RELATED_TO {source_id: $source_id, relation_text: $relation_text}]->(t)
        SET r.start_date = $start_date,
            r.end_date = $end_date,
            r.chunk_id = $chunk_id,
            r.relation_embedding = $relation_embedding
        """,
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        source_id=source_id,
        relation_text=relation_text,
        relation_embedding=relation_embedding,
        chunk_id=chunk_id,
        start_date=start_date,
        end_date=end_date,
    )


def embed_texts(
    texts: List[str], model: Optional[str] = None, expected_dim: Optional[int] = None
) -> List[List[float]]:
    api_key_env = "EMBEDDING_API_KEY" if os.getenv("EMBEDDING_API_KEY") else "MODEL_API_KEY"
    base_url_env = "EMBEDDING_BASE_URL" if os.getenv("EMBEDDING_BASE_URL") else "MODEL_BASE_URL"
    client = openai_client(api_key_env=api_key_env, base_url_env=base_url_env)
    model = model or EMBEDDING_MODEL
    expected_dim = expected_dim or EMBEDDING_DIM
    if not model:
        raise RuntimeError("EMBEDDING_MODEL is not set.")
    response = client.embeddings.create(model=model, input=texts)
    vectors = [item.embedding for item in response.data]
    for vec in vectors:
        if len(vec) != expected_dim:
            raise RuntimeError(
                f"Embedding dimension mismatch: expected {expected_dim}, got {len(vec)}"
            )
    return vectors


def ingest_text(text: str, source_id: Optional[str] = None) -> Dict[str, int]:
    chunks = chunk_text(text)

    print("got chunks:", len(chunks))

    if not chunks:
        return {"chunks": 0, "entities": 0, "relations": 0}

    embeddings = embed_texts(chunks)

    driver = _neo4j_driver()
    totals = {"chunks": 0, "entities": 0, "relations": 0}

    i = 0

    with driver.session() as session:
        for chunk, embedding in zip(chunks, embeddings):
            i += 1
            extracted_entities, extracted_relations = extract_entities_and_relations(chunk)

            if i % 5 == 0:
                print(f"made {i} llm calls")
                print(f"ingesting chunk {i}/{len(chunks)}: {len(extracted_entities)} entities, {len(extracted_relations)} relations")

            timestamp_ranges = {
                e.name: parse_timestamp_range(e.name)
                for e in extracted_entities
                if _is_time_entity(e.entity_type)
            }

            def ingest_chunk(tx):
                chunk_id = create_chunk(tx, chunk, embedding, source_id)
                entity_ids: Dict[str, str] = {}
                for entity in extracted_entities:
                    if _is_time_entity(entity.entity_type):
                        continue
                    entity_id = upsert_entity(tx, entity)
                    entity_ids[entity.name] = entity_id
                link_chunk_mentions(tx, chunk_id, entity_ids.values())

                relation_embeddings: List[List[float]] = []
                if extracted_relations:
                    relation_texts = [rel.description or "" for rel in extracted_relations]
                    relation_embeddings = embed_texts(relation_texts)
                relation_embedding_iter = iter(relation_embeddings)

                for rel in extracted_relations:
                    relation_embedding = next(relation_embedding_iter)
                    src_id = entity_ids.get(rel.source_entity)
                    tgt_id = entity_ids.get(rel.target_entity)
                    if not src_id or not tgt_id:
                        continue
                    tr = timestamp_ranges.get(rel.timestamp_entity, TimestampRange(None, None))
                    create_relationship(
                        tx,
                        source_id or chunk_id,
                        src_id,
                        tgt_id,
                        rel.description,
                        relation_embedding,
                        chunk_id,
                        tr.start_date,
                        tr.end_date,
                    )
                return chunk_id, len(entity_ids), len(extracted_relations)

            chunk_id, entity_count, rel_count = session.execute_write(ingest_chunk)
            totals["chunks"] += 1
            totals["entities"] += entity_count
            totals["relations"] += rel_count

    driver.close()
    return totals
