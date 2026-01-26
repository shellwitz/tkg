#!.venv/bin/python3
from collections import defaultdict
import json
import argparse
import logging
import subprocess
import time
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.answer_test import STOCK_CODES
from tkg_rag.logging_utils import setup_logging
from tkg_rag.ingest import ingest_text

logger = logging.getLogger(__name__)

def insert_simple(base_data):
    first_doc = json.loads(base_data[0])["raw_content"]

    logger.info("Ingesting text data")
    start_ts = time.time()
    output = ingest_text(first_doc, source_uri="CROCS/2020/Q1", source_last_modified=time.time())
    end_ts = time.time()
    logger.info("Ingestion time: %.2f seconds", end_ts - start_ts)
    logger.info("%s", output)

def insert_all(base_data):
    all_start_ts = time.time()
    for i, line in enumerate(base_data):
        info = json.loads(line)
        text = info["raw_content"]
        doc_uri = info["stock_code"] + "/" + info["year"] + "/" + info["quarter"]

        logger.info(
            "Ingesting text data for doc_id %s (%s/%s)",
            doc_uri,
            i + 1,
            len(base_data),
        )
        i_start_ts = time.time()
        output = ingest_text(text, source_uri=doc_uri, source_last_modified=time.time())
        i_end_ts = time.time()
        logger.info("Ingestion time: %.2f seconds", i_end_ts - i_start_ts)
        logger.info("%s", output)
    all_end_ts = time.time()
    logger.info("Total ingestion time for all documents: %.2f seconds", all_end_ts - all_start_ts)

def insert_wrt_q_stock_codes(base_data):
    all_start_ts = time.time()

    to_insert = []
    for line in base_data:
        info = json.loads(line)
        if info["stock_code"] in STOCK_CODES:
            to_insert.append(info)

    for i, entry in enumerate(to_insert):
        text = entry["raw_content"]
        doc_uri = entry["stock_code"] + "/" + entry["year"] + "/" + entry["quarter"]

        logger.info(
            "Ingesting text data for doc_id %s (%s/%s)",
            doc_uri,
            i + 1,
            len(to_insert),
        )
        i_start_ts = time.time()
        output = ingest_text(text, source_uri=doc_uri, source_last_modified=time.time())
        i_end_ts = time.time()
        logger.info("Ingestion time: %.2f seconds", i_end_ts - i_start_ts)
        logger.info("%s", output)
    all_end_ts = time.time()
    logger.info("Total ingestion time for question-index-based documents: %.2f seconds", all_end_ts - all_start_ts)
    
def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Ingest test data into Neo4j.")
    parser.add_argument("-f", "--fresh", action="store_true", help="Reset the environment.")
    parser.add_argument("-fb", "--fresh-build", action="store_true", help="Rebuild docker images and restart containers.")
    parser.add_argument("-q", "--question-stock-codes", action="store_true", help="Use predefined questions corresponding to predefined stock codes with solutions to then be able to compare RAG output to solutions.")
    parser.add_argument("-a", "--all", action="store_true", help="Ingest all documents in base.jsonl.")
    args = parser.parse_args()

    build = ""
    if args.fresh or args.fresh_build:
        if args.fresh_build:
            logger.info("Rebuilding docker images...")
            build = "--build"
        try:
            subprocess.run("docker compose down -v", shell=True, check=True)
            subprocess.run(f"docker compose up -d {build}", shell=True, check=True)
            # Give Neo4j and the schema entrypoint time to initialize
            logger.info("Waiting for Neo4j to initialize...")
            time.sleep(60) #damn sometimes takes long
        except subprocess.CalledProcessError as e:
            logger.error("Docker setup failed: %s", e)
            exit(1)
        logger.info("Docker containers are up and running.")

    with open("ect-qa/extracted/corpus/base.jsonl", "r") as f:
        base_data = f.readlines()

        if args.question_stock_codes:
            insert_wrt_q_stock_codes(base_data)
        elif args.all:
            insert_all(base_data)
        else:
            insert_simple(base_data)
    

if __name__ == "__main__":
    main()
