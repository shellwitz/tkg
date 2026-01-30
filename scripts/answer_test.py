#!.venv/bin/python3
import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tkg_rag.cypher_agent import run_cypher_agent
from tkg_rag.logging_utils import setup_logging
from tkg_rag.answer import generate_answer
from tkg_rag.retrieve import retrieve
import time

logger = logging.getLogger(__name__)


STOCK_CODES = {"EOG", 
               #"SKX", "EPAM US", "CINF",
               }

def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Run retrieval + answer generation.")
    parser.add_argument(
        "-q",
        "--question",
        default="What happened in 2020 Q1 related to Crocs?",
        help="User question to query the KG.",
    )
    parser.add_argument(
        "-qs",
        "--question-stock-codes",
        action="store_true",
        help="Use predefined questions for stock codes.",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Use cypher agent instead of RAG for question answering"
    )
    args = parser.parse_args()

    if args.question_stock_codes:
        ANSWER_PATH = "/home/shellwitz/Documents/uni_stuff/nlp_uni/tkg/eval/rag_results_to_evaluate/daniel_diy_tkg_big/vec_and_edge_search.jsonl"
        #"/home/shellwitz/Documents/uni_stuff/nlp_uni/tkg_eval/rag_results_to_evaluate/daniel_diy_tkg_big/vec_and_edge_search_less_context_tkg_answers.jsonl"

        with open("ect-qa/questions/local_base.jsonl", "r") as f:
            questions_raw = f.readlines()

        question_objs = []
        for q in questions_raw:
            question_obj = json.loads(q)
            are_all_evidence_codes_in_stock_codes = True


            for e in question_obj["evidence_list"]:
                if e["stock_code"] not in STOCK_CODES: 
                    are_all_evidence_codes_in_stock_codes = False
                    print(f"not all in evidence list {e["stock_code"]}")
                    break

            if are_all_evidence_codes_in_stock_codes and question_obj["answer"] != "unanswerable":
                question_objs.append(question_obj)


        with open(ANSWER_PATH, "w") as f:
            start_ts = time.time()
            for i, question_obj in enumerate(question_objs):
                if args.agent:
                    result = run_cypher_agent(
                        question=question_obj["question"],
                        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
                        neo4j_password=os.getenv("NEO4J_PASSWORD", "passworty"),
                        container=os.getenv("TKG_NEO4J_CONTAINER", "tkg-neo4j"),
                        model=os.getenv("AGENT_MODEL"),
                        timeout_s=float(os.getenv("AGENT_TIMEOUT", "15.0")),
                        max_steps=int(os.getenv("AGENT_MAX_STEPS", "20")),
                    )
                    question_obj["predicted_answer"] = result.get("answer", "")
                    #question_obj["context"] = json.dumps(result, indent=2)
                else:
                    result = retrieve(question_obj["question"])
                    answer = generate_answer(result["question"], result["context"])
                    logger.info("generated answer: %s/%s", i + 1, len(question_objs))
                    question_obj["predicted_answer"]  = answer
                    question_obj["context"] = result["context"]

                f.write(json.dumps(question_obj) + "\n")
            end_ts = time.time()
            elapsed = end_ts - start_ts
            logger.info("RAG questions eval from question indices took time: %.2f seconds", elapsed)
    else:
        if args.agent:
            result = run_cypher_agent(
                question=args.question,
                neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
                neo4j_password=os.getenv("NEO4J_PASSWORD", "passworty"),
                container=os.getenv("TKG_NEO4J_CONTAINER", "tkg-neo4j"),
                model=os.getenv("AGENT_MODEL"),
                timeout_s=float(os.getenv("AGENT_TIMEOUT", "15.0")),
                max_steps=int(os.getenv("AGENT_MAX_STEPS", "15")),
            )
            logger.info("Agent Answer:\n%s", result.get("answer", ""))
            logger.info("Cypher Query:\n%s", result.get("cypher", ""))
        else:
            payload = retrieve(args.question)
            logger.info("Context:\n%s", payload["context"])
            logger.info("Answer:\n%s", generate_answer(args.question, payload["context"]))


if __name__ == "__main__":
    main()
