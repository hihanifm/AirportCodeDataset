#!/bin/bash
# Stop the enrichment script and its caffeinate wrapper.

pkill -f "caffeinate.*enrich_airport_codes" 2>/dev/null
pkill -f "python.*enrich_airport_codes" 2>/dev/null

if pgrep -f "enrich_airport_codes" > /dev/null 2>&1; then
    echo "Warning: process still running"
    pgrep -fl "enrich_airport_codes"
else
    echo "Stopped."
fi
