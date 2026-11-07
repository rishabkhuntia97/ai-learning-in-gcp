#!/bin/bash
# Usage: bash scripts/end_day.sh   (save your notebooks first!)
set -e
cd "$(git rev-parse --show-toplevel)"

TODAY_DIR=$(ls -d days/$(date +%F)-* 2>/dev/null | head -1 || true)
TOPIC="${TODAY_DIR#days/????-??-??-}"

# Add today to the README learning log
if [ -n "$TODAY_DIR" ] && ! grep -q "($TODAY_DIR)" README.md; then
  echo "| $(date +%F) | $TOPIC | [open]($TODAY_DIR) |" >> README.md
fi

git add -A
git commit -m "Day $(date +%F): ${TOPIC:-updates}" || echo "Nothing new to commit."
git push
echo ""
echo "Pushed. Open GitHub and check today's folder is there, then it's safe to delete the instance."
