import json
import os
import subprocess
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from neo4j import READ_ACCESS, GraphDatabase
from neo4j.graph import Node, Path, Relationship
from neo4j.exceptions import Neo4jError

from . import prompts
from .ingest import embed_texts
from .llm_client import openai_client
from .settings import LLM_MODEL


def load_effective_schema_from_container(
    container: str | None = None,
    timeout_s: float = 5.0,
) -> str:
    """Load the effective schema text (including runtime vector indexes) from Neo4j container."""
    target = container or os.getenv("TKG_NEO4J_CONTAINER", "tkg-neo4j")
    try:
        raw = subprocess.check_output(
            ["docker", "exec", target, "cat", "/tmp/schema.cypher"],
            timeout=timeout_s,
        )
        return raw.decode("utf-8", errors="replace").strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        raw = subprocess.check_output(
            ["docker", "exec", target, "cat", "/init/schema.cypher"],
            timeout=timeout_s,
        )
        return raw.decode("utf-8", errors="replace").strip()


def _run_introspection_query(session, cypher: str, timeout_s: float) -> List[Dict[str, Any]]:
    result = session.run(cypher, timeout=timeout_s)
    return [record.data() for record in result]


def fetch_db_introspection(driver, timeout_s: float = 5.0) -> Dict[str, Any]:
    """Collect labels, relationship types, keys, indexes, and constraints for prompt grounding."""
    with driver.session(default_access_mode=READ_ACCESS) as session:
        labels = [row["label"] for row in _run_introspection_query(session, "CALL db.labels()", timeout_s)]
        relationship_types = [
            row["relationshipType"]
            for row in _run_introspection_query(session, "CALL db.relationshipTypes()", timeout_s)
        ]
        property_keys = [
            row["propertyKey"] for row in _run_introspection_query(session, "CALL db.propertyKeys()", timeout_s)
        ]
        indexes = _run_introspection_query(
            session,
            "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, state",
            timeout_s,
        )
        constraints = _run_introspection_query(
            session,
            "SHOW CONSTRAINTS YIELD name, type, entityType, labelsOrTypes, properties",
            timeout_s,
        )
    return {
        "labels": sorted(labels),
        "relationship_types": sorted(relationship_types),
        "property_keys": sorted(property_keys),
        "indexes": indexes,
        "constraints": constraints,
    }


def _format_introspection(introspection: Dict[str, Any]) -> Dict[str, str]:
    return {
        "labels": ", ".join(introspection["labels"]),
        "relationship_types": ", ".join(introspection["relationship_types"]),
        "property_keys": ", ".join(introspection["property_keys"]),
        "indexes": json.dumps(introspection["indexes"], indent=2, ensure_ascii=True),
        "constraints": json.dumps(introspection["constraints"], indent=2, ensure_ascii=True),
    }


def _build_system_prompt(schema_text: str, introspection: Dict[str, Any]) -> str:
    formatted = _format_introspection(introspection)
    return prompts.CYPHER_AGENT_SYS_PROMPT.format(
        schema_cypher=schema_text,
        labels=formatted["labels"],
        relationship_types=formatted["relationship_types"],
        property_keys=formatted["property_keys"],
        indexes=formatted["indexes"],
        constraints=formatted["constraints"],
    )


def run_readonly_query(
    driver,
    cypher: str,
    parameters: Dict[str, Any] | None = None,
    timeout_s: float = 15.0,
    get_embedding: Optional[Callable[[], List[float]]] = None,
) -> List[Dict[str, Any]]:
    """Execute a read-only query with optional lazy embedding injection."""
    params = dict(parameters or {})
    # Lazy embedding: only compute if the query uses $question_embedding
    if get_embedding is not None and "$question_embedding" in cypher:
        params["question_embedding"] = get_embedding()
    with driver.session(default_access_mode=READ_ACCESS) as session:
        result = session.run(cypher, params, timeout=timeout_s)
        return [record.data() for record in result]


class _Neo4jEncoder(json.JSONEncoder):
    """Handle Neo4j temporal types and other non-JSON-serializable objects."""

    def default(self, o: Any) -> Any:
        if isinstance(o, Node):
            return {
                "_type": "Node",
                "id": o.id,
                "labels": list(o.labels),
                "properties": dict(o),
            }
        if isinstance(o, Relationship):
            return {
                "_type": "Relationship",
                "id": o.id,
                "type": o.type,
                "properties": dict(o),
            }
        if isinstance(o, Path):
            return {
                "_type": "Path",
                "nodes": list(o.nodes),
                "relationships": list(o.relationships),
            }
        # neo4j.time.Date, neo4j.time.DateTime, etc.
        if hasattr(o, "iso_format"):
            return o.iso_format()
        if hasattr(o, "isoformat"):
            return o.isoformat()
        return super().default(o)


def _normalize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return json.loads(json.dumps(rows, cls=_Neo4jEncoder))


def _log_event(log_path: str | None, event: Dict[str, Any]) -> None:
    if not log_path:
        return
    event_type = event.get("event", "event").upper()
    ts = datetime.utcnow().isoformat() + "Z"
    header = f"{event_type} [{ts}]"
    line = "-" * max(20, len(header))
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"{header}\n{line}\n")
        if event_type == "QUESTION":
            handle.write(f"{event.get('question', '')}\n\n")
        elif event_type == "LLM_OUTPUT":
            handle.write(f"{event.get('content', '')}\n\n")
        elif event_type == "CYPHER_RESULT":
            handle.write(json.dumps(event.get("rows", []), indent=2, ensure_ascii=True, cls=_Neo4jEncoder) + "\n\n")
        else:
            handle.write(json.dumps(event, indent=2, ensure_ascii=True, cls=_Neo4jEncoder) + "\n\n")


def run_cypher_agent(
    question: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    container: str | None = None,
    model: str | None = None,
    timeout_s: float = 15.0,
    max_steps: int = 5,
    log_path: str | None = None,
) -> Dict[str, Any]:
    """Run the LLM-driven Cypher agent until it returns a FINAL answer or max steps is reached."""
    if not (model or LLM_MODEL):
        raise RuntimeError("LLM_MODEL is not set.")
    schema_text = load_effective_schema_from_container(container=container)
    client = openai_client()
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    # Lazy embedding: compute only on first use
    _cached_embedding: List[float] | None = None

    def get_question_embedding() -> List[float]:
        nonlocal _cached_embedding
        if _cached_embedding is None:
            _cached_embedding = embed_texts([question])[0]
        return _cached_embedding

    try:
        introspection = fetch_db_introspection(driver, timeout_s=min(timeout_s, 5.0))
        system_prompt = _build_system_prompt(schema_text, introspection)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompts.CYPHER_AGENT_QUERY_PROMPT.format(question=question)},
        ]
        _log_event(log_path, {"event": "question", "question": question})
        last_cypher = ""
        last_rows: List[Dict[str, Any]] = []
        steps: List[Dict[str, Any]] = []
        for _ in range(max_steps):
            response = client.chat.completions.create(
                model=model or LLM_MODEL,
                messages=messages,
                temperature=0,
            )
            content = (response.choices[0].message.content or "").strip()
            _log_event(log_path, {"event": "llm_output", "content": content})
            if content.startswith("FINAL:"):
                return {
                    "answer": content[len("FINAL:") :].strip(),
                    "cypher": last_cypher,
                    "rows": _normalize_rows(last_rows),
                    "steps": steps,
                }
            
            if not content.startswith("QUERY:"):
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": "Your output has to either start with 'QUERY:' or 'FINAL:'"
                    }
                )
                continue

            cypher = content[len("QUERY:") :].strip()
            last_cypher = cypher
            try:
                last_rows = run_readonly_query(driver, cypher, timeout_s=timeout_s, get_embedding=get_question_embedding)
            except Neo4jError as exc:
                last_rows = [{"__error__": str(exc)}]
            normalized_rows = _normalize_rows(last_rows)
            steps.append({"cypher": cypher, "rows": normalized_rows})
            _log_event(log_path, {"event": "cypher_result", "rows": last_rows})
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": prompts.CYPHER_AGENT_OBSERVATION_PROMPT.format(
                        cypher=cypher,
                        results=json.dumps(last_rows, indent=2, ensure_ascii=True, cls=_Neo4jEncoder),
                    ),
                }
            )
        messages.append(
            {
                "role": "user",
                "content": f"""Just give an answer with the available information you got now.
                Question: {question}"""
            }
        )

        response = client.chat.completions.create(
                model=model or LLM_MODEL,
                messages=messages,
                temperature=0,
            )
        content = (response.choices[0].message.content or "").strip()
        return {
            "answer": content,
            "cypher": last_cypher,
            "rows": _normalize_rows(last_rows),
            "steps": steps,
        }
    finally:
        driver.close()


def run_cypher_agent_stream(
    question: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    container: str | None = None,
    model: str | None = None,
    timeout_s: float = 15.0,
    max_steps: int = 5,
    log_path: str | None = None,
):
    if not (model or LLM_MODEL):
        raise RuntimeError("LLM_MODEL is not set.")
    schema_text = load_effective_schema_from_container(container=container)
    client = openai_client()
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    _cached_embedding: List[float] | None = None

    def get_question_embedding() -> List[float]:
        nonlocal _cached_embedding
        if _cached_embedding is None:
            _cached_embedding = embed_texts([question])[0]
        return _cached_embedding

    try:
        introspection = fetch_db_introspection(driver, timeout_s=min(timeout_s, 5.0))
        system_prompt = _build_system_prompt(schema_text, introspection)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompts.CYPHER_AGENT_QUERY_PROMPT.format(question=question)},
        ]
        _log_event(log_path, {"event": "question", "question": question})
        last_cypher = ""
        last_rows: List[Dict[str, Any]] = []
        steps: List[Dict[str, Any]] = []
        for _ in range(max_steps):
            response = client.chat.completions.create(
                model=model or LLM_MODEL,
                messages=messages,
                temperature=0,
            )
            content = (response.choices[0].message.content or "").strip()
            _log_event(log_path, {"event": "llm_output", "content": content})
            if content.startswith("FINAL:"):
                final_answer = content[len("FINAL:") :].strip()
                yield {
                    "event": "final",
                    "answer": final_answer,
                    "cypher": last_cypher,
                    "rows": _normalize_rows(last_rows),
                    "steps": steps,
                }
                return

            if not content.startswith("QUERY:"):
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": "Your output has to either start with 'QUERY:' or 'FINAL:'",
                    }
                )
                continue

            cypher = content[len("QUERY:") :].strip()
            last_cypher = cypher
            try:
                last_rows = run_readonly_query(
                    driver,
                    cypher,
                    timeout_s=timeout_s,
                    get_embedding=get_question_embedding,
                )
            except Neo4jError as exc:
                last_rows = [{"__error__": str(exc)}]
            normalized_rows = _normalize_rows(last_rows)
            step = {"cypher": cypher, "rows": normalized_rows}
            steps.append(step)
            _log_event(log_path, {"event": "cypher_result", "rows": last_rows})
            yield {"event": "step", **step}
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": prompts.CYPHER_AGENT_OBSERVATION_PROMPT.format(
                        cypher=cypher,
                        results=json.dumps(last_rows, indent=2, ensure_ascii=True, cls=_Neo4jEncoder),
                    ),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": f"""Just give an answer with the available information you got now.
                Question: {question}""",
            }
        )
        response = client.chat.completions.create(
            model=model or LLM_MODEL,
            messages=messages,
            temperature=0,
        )
        content = (response.choices[0].message.content or "").strip()
        yield {
            "event": "final",
            "answer": content,
            "cypher": last_cypher,
            "rows": _normalize_rows(last_rows),
            "steps": steps,
        }
    finally:
        driver.close()
