#!/bin/bash

PROJECT_DIR="$HOME/Documents/uni_stuff/nlp_uni/tkg"

systemd-inhibit --why="overnight tk rag eval" --mode=block bash -lc \
"source $PROJECT_DIR/.venv/bin/activate && python $PROJECT_DIR/eval/eval_basic.py"