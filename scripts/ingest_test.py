#!.venv/bin/python3
from collections import defaultdict
import json
import argparse
import subprocess
import time
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tkg_rag.ingest import ingest_text

def insert_simple(base_data):
    first_doc = json.loads(base_data[0])["raw_content"]

    print("Ingesting text data")
    output = ingest_text(first_doc, "CROCS_2020_q1")
    print(output)


QUESTION_INDICES = [4, 5, 6,7, 8]

def insert_wrt_q_indices(base_data):

    with open("ect-qa/questions/local_base.jsonl", "r") as f:
        questions = f.readlines()

    base_data_stock_code_map = defaultdict(list)

    for line in base_data:
        info = json.loads(line)
        base_data_stock_code_map[info["stock_code"]].append(info)

    to_insert = []

    question_objs = []

    for i in QUESTION_INDICES:
        question_obj = json.loads(questions[i])
        evidence_list = question_obj["evidence_list"]
        question_objs.append(question_obj)

        if not evidence_list: #unanswerable question
            continue
        stock_codes = {e["stock_code"] for e in evidence_list} #no clue if there actually exist more than one stock codes for evidence
        for stock_code in stock_codes:
            to_insert.extend(base_data_stock_code_map[stock_code])
    
    for entry in to_insert:
        text = entry["raw_content"]
        doc_id = entry["stock_code"] + "_" + entry["year"] + "_" + entry["quarter"]
        print(f"Ingesting text data for doc_id {doc_id}")
        output = ingest_text(text, doc_id)
        print(output)

def main():
    parser = argparse.ArgumentParser(description="Ingest test data into Neo4j.")
    parser.add_argument("-f", "--fresh", action="store_true", help="Reset the environment.")
    parser.add_argument("-fb", "--fresh-build", action="store_true", help="Rebuild docker images and restart containers.")
    parser.add_argument("-q", "--question-indices", action="store_true", help="Use 5 questions with solutions to then be able to compare RAG output to solutions.")
    args = parser.parse_args()

    build = ""
    if args.fresh or args.fresh_build:
        if args.fresh_build:
            print("Rebuilding docker images...")
            build = "--build"
        try:
            subprocess.run("docker compose down -v", shell=True, check=True)
            subprocess.run(f"docker compose up -d {build}", shell=True, check=True)
            # Give Neo4j and the schema entrypoint time to initialize
            print("Waiting for Neo4j to initialize...")
            time.sleep(60) #damn sometimes takes long
        except subprocess.CalledProcessError as e:
            print(f"Docker setup failed: {e}")
            exit(1)
        print("Docker containers are up and running.")

    with open("ect-qa/extracted/corpus/base.jsonl", "r") as f:
        base_data = f.readlines()

        if args.question_indices:
            insert_wrt_q_indices(base_data)
        else:
            insert_simple(base_data)
    

if __name__ == "__main__":
    main()
