#!.venv/bin/python3
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tkg_rag.answer import generate_answer
from tkg_rag.retrieve import retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval + answer generation.")
    parser.add_argument(
        "-q",
        "--question",
        default="What happened in 2020 Q1 related to Crocs?",
        help="User question to query the KG.",
    )
    parser.add_argument(
        "-k",
        "--max-edges",
        type=int,
        default=25,
        help="Max edges to return from retrieval.",
    )
    args = parser.parse_args()

    payload = retrieve(args.question, max_edges=args.max_edges)
    print("Context:\n")
    print(payload["context"])
    print("\nAnswer:\n")
    print(generate_answer(args.question, payload["context"]))


if __name__ == "__main__":
    main()
