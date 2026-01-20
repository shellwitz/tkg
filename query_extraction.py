from dataclasses import dataclass
from typing import Dict, List, Tuple

import prompts
from llm_client import openai_client
from settings import ENTITY_TYPES, LLM_MODEL


DEFAULT_QUERY_TIME_TYPES = ["date", "date_range", "quarter", "year"]
DEFAULT_TIMESTAMP_FORMAT = "ISO-8601 or ISO-like (YYYY, YYYY-MM-DD, YYYY-Qn)"


@dataclass
class QueryEntity:
    name: str
    entity_type: str


def is_time_entity(entity_type: str) -> bool:
    return entity_type in DEFAULT_QUERY_TIME_TYPES or entity_type == "timestamp"


def _build_query_prompts() -> Tuple[str, str, Dict[str, str]]:
    tuple_delimiter = "|"
    record_delimiter = ";;"
    system_prompt = prompts.QUERY_ENTITY_TIME_EXTRACTION_SYS_PROMPT.format(
        entity_types=", ".join(ENTITY_TYPES),
        tuple_delimiter=tuple_delimiter,
        record_delimiter=record_delimiter,
        timestamp_types=", ".join(DEFAULT_QUERY_TIME_TYPES),
        timestamp_format=DEFAULT_TIMESTAMP_FORMAT,
    )
    user_prompt = prompts.QUERY_ENTITY_TIME_EXTRACTION_USER_PROMPT
    return system_prompt, user_prompt, {
        "tuple_delimiter": tuple_delimiter,
        "record_delimiter": record_delimiter,
    }


def _parse_query_output(raw: str, tuple_delimiter: str, record_delimiter: str) -> List[QueryEntity]:
    entities: List[QueryEntity] = []
    for record in raw.split(record_delimiter):
        record = record.strip().strip(",")
        if not record:
            continue
        record = record.strip("() ")
        parts = [p.strip().strip('"') for p in record.split(tuple_delimiter)]
        if len(parts) < 3:
            continue
        if parts[0] == "entity":
            entities.append(QueryEntity(parts[1], parts[2]))
    return entities


def extract_query_entities(question: str) -> List[QueryEntity]:
    if not LLM_MODEL:
        raise RuntimeError("LLM_MODEL is not set.")
    system_prompt, user_prompt_template, delimiters = _build_query_prompts()
    client = openai_client()
    user_prompt = user_prompt_template.format(entity_types=", ".join(ENTITY_TYPES), question=question)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content or ""
    return _parse_query_output(raw, delimiters["tuple_delimiter"], delimiters["record_delimiter"])
