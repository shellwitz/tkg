#!/bin/bash

systemd-inhibit --why="overnight TKG evaluation" --mode=block bash -lc '
    scripts/run_overnight_ingestion.sh > /dev/null
    scripts/run_overnight_answering.sh > /dev/null
    eval/run_overnight_eval.sh > /dev/null'