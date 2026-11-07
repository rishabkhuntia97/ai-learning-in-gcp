#!/bin/bash
# Usage: bash scripts/start_day.sh "topic-name"
set -e
TOPIC="${1:?Usage: bash scripts/start_day.sh <topic-name>}"
TOPIC=$(echo "$TOPIC" | tr ' ' '-' | tr '[:upper:]' '[:lower:]')

# Fresh instance each day, so set git identity.
# Uses GitHub's private noreply email so your real email never appears in this public repo.
git config --global user.name  "Rishab Khuntia"
git config --global user.email "118143011+rishabkhuntia97@users.noreply.github.com"
git config --global credential.helper 'cache --timeout=43200'   # remember token for 12h

cd "$(git rev-parse --show-toplevel)"
git pull --ff-only || true

DIR="days/$(date +%F)-$TOPIC"
mkdir -p "$DIR"
if [ ! -f "$DIR/notes.md" ]; then
  cat > "$DIR/notes.md" <<NOTES
# $(date +%F) — $TOPIC

## What I learned

## Commands / code worth remembering

## Gotchas

## Next time
NOTES
fi
echo "Today's folder: $DIR"
echo "Create your notebooks there. Run 'bash scripts/end_day.sh' before deleting the instance."
