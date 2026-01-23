#!/bin/bash

PROJECT_DIR="$HOME/Documents/uni_stuff/nlp_uni/tkg"

systemd-inhibit --why="overnight tk rag insert" --mode=block bash -lc \
"source $PROJECT_DIR/.venv/bin/activate && python $PROJECT_DIR/scripts/ingest_test.py -f -a"