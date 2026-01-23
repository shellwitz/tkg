import os

ENTITY_TYPES = [
    "financial concept",
    "business segment",
    "event",
    "company",
    "person",
    "product",
    "location",
    "organization",
]

LLM_MODEL = os.getenv("LLM_MODEL", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "4096"))
RELATION_DEDUP_SIM_THRESHOLD = float(os.getenv("RELATION_DEDUP_SIM_THRESHOLD", "0.85"))
