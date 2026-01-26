#!/bin/bash

systemd-inhibit --why="overnight vibe coding" --mode=block bash -lc \
"codex exec \"Do the steps in the AGENTS.md\""