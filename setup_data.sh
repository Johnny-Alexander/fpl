#!/usr/bin/env bash
# Fetch the historical dataset into data/historical.
#
# A full clone of vaastav/Fantasy-Premier-League is ~333MB across every season
# since 2016-17. The model only reads the seasons listed below, so this pulls just
# those -- roughly 57MB for three seasons, which makes the project practical to
# set up in a fresh cloud environment or over a phone connection.
#
#   ./setup_data.sh                        # default seasons
#   ./setup_data.sh 2023-24 2024-25 ...    # override
#
# Uses git sparse-checkout where available (git >= 2.25) and falls back to the
# legacy core.sparseCheckout mechanism on older git.

set -euo pipefail

REPO="https://github.com/vaastav/Fantasy-Premier-League.git"
BRANCH="master"
DEST="$(cd "$(dirname "$0")" && pwd)/data/historical"

SEASONS=("$@")
if [ ${#SEASONS[@]} -eq 0 ]; then
  SEASONS=(2024-25 2025-26 2026-27)
fi

git_at_least() {
  local want=$1
  local have
  have=$(git --version | awk '{print $3}')
  [ "$(printf '%s\n%s\n' "$want" "$have" | sort -V | head -1)" = "$want" ]
}

write_legacy_sparse_config() {
  git -C "$DEST" config core.sparseCheckout true
  : > "$DEST/.git/info/sparse-checkout"
  for season in "${SEASONS[@]}"; do
    echo "data/$season/*" >> "$DEST/.git/info/sparse-checkout"
  done
}

if [ -d "$DEST/.git" ]; then
  echo "Updating existing dataset in $DEST"
  if git_at_least 2.25; then
    paths=(); for s in "${SEASONS[@]}"; do paths+=("data/$s"); done
    git -C "$DEST" sparse-checkout set "${paths[@]}"
  else
    write_legacy_sparse_config
  fi
  git -C "$DEST" fetch --depth 1 origin "$BRANCH"
  git -C "$DEST" reset --hard FETCH_HEAD
else
  echo "Fetching ${#SEASONS[@]} season(s) into $DEST"
  mkdir -p "$DEST"
  git init -q "$DEST"
  git -C "$DEST" remote add origin "$REPO"

  if git_at_least 2.25; then
    git -C "$DEST" sparse-checkout init --cone
    paths=(); for s in "${SEASONS[@]}"; do paths+=("data/$s"); done
    git -C "$DEST" sparse-checkout set "${paths[@]}"
  else
    write_legacy_sparse_config
  fi

  # Blobless fetch keeps the transfer small; harmless to omit on older git.
  if git_at_least 2.19; then
    git -C "$DEST" fetch --depth 1 --filter=blob:none origin "$BRANCH"
  else
    git -C "$DEST" fetch --depth 1 origin "$BRANCH"
  fi
  git -C "$DEST" checkout -q FETCH_HEAD
fi

echo
missing=0
for season in "${SEASONS[@]}"; do
  file="$DEST/data/$season/gws/merged_gw.csv"
  if [ -f "$file" ]; then
    printf '  %s  ok   (%s gameweek rows)\n' "$season" "$(($(wc -l < "$file") - 1))"
  else
    printf '  %s  MISSING - not published upstream yet\n' "$season"
    missing=1
  fi
done

echo
echo "Done. $(du -sh "$DEST" | cut -f1) on disk."
[ "$missing" -eq 0 ] || echo "Note: a missing current season is normal early on; the tool fills finished gameweeks from the API."
