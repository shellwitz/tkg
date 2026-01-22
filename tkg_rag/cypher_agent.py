import json
import os
import subprocess
from datetime import datetime
from typing import Any, Dict, List

from neo4j import READ_ACCESS, GraphDatabase
from neo4j.exceptions import Neo4jError

from . import prompts
from .llm_client import openai_client
from .settings import LLM_MODEL


def load_effective_schema_from_container(
    container: str | None = None,
    timeout_s: float = 5.0,
) -> str:
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
) -> List[Dict[str, Any]]:
    with driver.session(default_access_mode=READ_ACCESS) as session:
        result = session.run(cypher, parameters or {}, timeout=timeout_s)
        return [record.data() for record in result]


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
            handle.write(json.dumps(event.get("rows", []), indent=2, ensure_ascii=True) + "\n\n")
        else:
            handle.write(json.dumps(event, indent=2, ensure_ascii=True) + "\n\n")


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
    if not (model or LLM_MODEL):
        raise RuntimeError("LLM_MODEL is not set.")
    schema_text = load_effective_schema_from_container(container=container)
    client = openai_client()
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
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
                    "rows": last_rows,
                }
            if not content.startswith("QUERY:"):
                raise RuntimeError(f"Unexpected agent response: {content[:200]}")
            cypher = content[len("QUERY:") :].strip()
            last_cypher = cypher
            try:
                last_rows = run_readonly_query(driver, cypher, timeout_s=timeout_s)
            except Neo4jError as exc:
                last_rows = [{"__error__": str(exc)}]
            _log_event(log_path, {"event": "cypher_result", "rows": last_rows})
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": prompts.CYPHER_AGENT_OBSERVATION_PROMPT.format(
                        cypher=cypher,
                        results=json.dumps(last_rows, indent=2, ensure_ascii=True),
                    ),
                }
            )
        raise RuntimeError("Agent did not produce FINAL within max_steps.")
    finally:
        driver.close()
