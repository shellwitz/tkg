#!.venv/bin/python3
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tkg_rag.cypher_agent import run_cypher_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Cypher agent against Neo4j.")
    parser.add_argument(
        "-q",
        "--question",
        default="What happened in 2020 Q1 related to Crocs?",
        help="User question to query the KG.",
    )
    parser.add_argument(
        "--uri",
        default=os.getenv("TKG_NEO4J_URI", "bolt://localhost:7688"),
        help="Neo4j Bolt URI.",
    )
    parser.add_argument(
        "--user",
        default=os.getenv("TKG_READONLY_USER", "tkg_reader"),
        help="Read-only Neo4j user.",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("TKG_READONLY_PASSWORD", "tkg_reader_pass"),
        help="Read-only Neo4j password.",
    )
    parser.add_argument(
        "--container",
        default=os.getenv("TKG_NEO4J_CONTAINER", "tkg-neo4j"),
        help="Neo4j container name for schema extraction.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(os.getenv("TKG_CYPHER_AGENT_MAX_STEPS", "5")),
        help="Maximum agent steps before failing.",
    )
    default_log = str((Path(__file__).resolve().parent / "logs" / "cypher_agent.log"))
    parser.add_argument(
        "--log-file",
        default=default_log,
        help="Append JSONL logs for LLM output and Cypher results.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("TKG_CYPHER_AGENT_TIMEOUT", "15")),
        help="Per-query timeout in seconds.",
    )
    args = parser.parse_args()

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = run_cypher_agent(
        question=args.question,
        neo4j_uri=args.uri,
        neo4j_user=args.user,
        neo4j_password=args.password,
        container=args.container,
        timeout_s=args.timeout,
        max_steps=args.max_steps,
        log_path=str(log_path),
    )
    print(result["answer"])


if __name__ == "__main__":
    main()
