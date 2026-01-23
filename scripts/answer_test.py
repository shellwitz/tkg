#!.venv/bin/python3
import argparse
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tkg_rag.logging_utils import setup_logging
from tkg_rag.answer import generate_answer
from tkg_rag.retrieve import retrieve

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
    args = parser.parse_args()

    payload = retrieve(args.question)
    logger.info("Context:\n%s", payload["context"])
    logger.info("Answer:\n%s", generate_answer(args.question, payload["context"]))


if __name__ == "__main__":
    main()
