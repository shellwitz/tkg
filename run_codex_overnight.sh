#!/bin/bash

systemd-inhibit --why="overnight vibe coding" --mode=block bash -lc \
"while true; do codex exec \"Do the steps in the AGENTS.md\"; done"