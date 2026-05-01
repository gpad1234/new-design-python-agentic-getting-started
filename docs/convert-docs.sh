#!/usr/bin/env bash
# convert-docs.sh — Convert Markdown docs in this directory to HTML using npx marked
#
# USAGE:
#   ./convert-docs.sh              # converts all .md files in this folder
#   ./convert-docs.sh <file.md>    # converts one or more specific files
#
# REQUIREMENTS:
#   Node.js (uses npx — no global install needed)
#
# OUTPUT:
#   HTML files are written next to their source .md files with the same name.
#   e.g. docs/spec_base_agent.md -> docs/spec_base_agent.html

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

convert() {
  local src="$1"
  local dest="${src%.md}.html"
  npx --yes marked "$src" -o "$dest"
  echo "Converted: $src -> $dest"
}

if [ $# -gt 0 ]; then
  for file in "$@"; do
    convert "$file"
  done
else
  shopt -s nullglob
  files=("$SCRIPT_DIR"/*.md)
  shopt -u nullglob

  if [ ${#files[@]} -eq 0 ]; then
    echo "No markdown files found in $SCRIPT_DIR"
    exit 0
  fi

  for f in "${files[@]}"; do
    convert "$f"
  done
fi
