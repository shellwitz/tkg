#!.venv/bin/python3
import json
import argparse
import subprocess
import time
from dotenv import load_dotenv

load_dotenv()

from ingest import ingest_text

def main():
    parser = argparse.ArgumentParser(description="Ingest test data into Neo4j.")
    parser.add_argument("-f", "--fresh", action="store_true", help="Reset the environment.")
    parser.add_argument("-fb", "--fresh-build", action="store_true", help="Rebuild docker images and restart containers.")
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
            time.sleep(15)
        except subprocess.CalledProcessError as e:
            print(f"Docker setup failed: {e}")
            exit(1)
        print("Docker containers are up and running.")

    with open("ect-qa/extracted/corpus/base.jsonl", "r") as f:
        base_data = f.readlines()

    first_doc = json.loads(base_data[0])["raw_content"]

    print("Ingesting text data")
    output = ingest_text(first_doc, "CROCS_2020_q1")
    print(output)

if __name__ == "__main__":
    main()
