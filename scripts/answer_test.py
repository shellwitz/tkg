#!.venv/bin/python3
import argparse
import json
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ingest_test import QUESTION_INDICES
from tkg_rag.logging_utils import setup_logging
from tkg_rag.answer import generate_answer
from tkg_rag.retrieve import retrieve
import time

logger = logging.getLogger(__name__)


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
        "-qi",
        "--question-indices",
        action="store_true",
        help="Use predefined question indices from ingest_test.",
    )
    args = parser.parse_args()

    if args.question_indices:
        ANSWER_PATH = "/home/shellwitz/Documents/uni_stuff/nlp_uni/tkg_eval/rag_results_to_evaluate/daniel_diy_tkg/vec_search_tkg_answers.jsonl"

        with open("ect-qa/questions/local_base.jsonl", "r") as f:
            questions_raw = f.readlines()


        with open(ANSWER_PATH, "a") as f:
            start_ts = time.time()
            for q_i in QUESTION_INDICES:
                question_obj = json.loads(questions_raw[q_i])
                result = retrieve(question_obj["question"])
                answer = generate_answer(result["question"], result["context"])
                question_obj["predicted_answer"]  = answer
                question_obj["context"] = result["context"]
                logger.info("Question %d answer: %s", q_i, answer)
                f.write(json.dumps(question_obj) + "\n")
            end_ts = time.time()
            elapsed = end_ts - start_ts
            logger.info("RAG questions eval from question indices took time: %.2f seconds", elapsed)
    else:
        payload = retrieve(args.question)
        logger.info("Context:\n%s", payload["context"])
        logger.info("Answer:\n%s", generate_answer(args.question, payload["context"]))


if __name__ == "__main__":
    main()
