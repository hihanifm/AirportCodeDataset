#!/bin/bash
# Run enrichment in background; keeps running when Mac screen is off.
# Uses caffeinate to prevent system sleep, nohup to survive terminal close.
#
# Usage:
#   ./run_enrich.sh                          # OpenAI (default)
#   ./run_enrich.sh --provider gemini        # Gemini
#   ./run_enrich.sh --provider gemini --model gemini-1.5-pro

cd "$(dirname "$0")"
source .venv/bin/activate || exit 1

LOG="enrich.log"
echo "Starting enrichment (log: $LOG). Use: tail -f $LOG to watch."
echo "Args: $@"
nohup caffeinate -i python enrich_airport_codes.py "$@" > "$LOG" 2>&1 &
echo "PID: $!"
