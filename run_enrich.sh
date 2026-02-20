#!/bin/bash
# Run enrichment in background; keeps running when Mac screen is off.
# Uses caffeinate to prevent system sleep, nohup to survive terminal close.

cd "$(dirname "$0")"
source .venv/bin/activate || exit 1

LOG="${1:-enrich.log}"
echo "Starting enrichment (log: $LOG). Use: tail -f $LOG to watch."
nohup caffeinate -i python enrich_airport_codes.py >> "$LOG" 2>&1 &
echo "PID: $!"
