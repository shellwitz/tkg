import json

from dotenv import load_dotenv

load_dotenv()

from ingest import ingest_text

with open("ect-qa/extracted/corpus/base.jsonl", "r") as f:
    base_data = f.readlines()

first_doc = json.loads(base_data[0])["raw_content"]

output = ingest_text(first_doc, "CROCS_2020_q1")
print(output)
