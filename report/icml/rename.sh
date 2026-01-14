#!/usr/bin/env bash
# Works in both bash and zsh

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "❌ Usage: $0 NewPaperName (e.g. my_icml_paper)"
  exit 1
fi

OLD="titleofreport"
NEW="$1"

echo "🔁 Renaming files and references: $OLD → $NEW"

# Step 1: Rename files
for file in ./*; do
  basename=$(basename "$file")
  if [[ "$basename" == *"$OLD"* ]]; then
    newname="${basename//$OLD/$NEW}"
    echo "📁 Renaming $basename → $newname"
    mv "$basename" "$newname"
  fi
done

# Step 2: Replace contents inside relevant files
for ext in tex bib sh; do
  for file in *.$ext; do
    if grep -q "$OLD" "$file"; then
      echo "✏️  Updating references in $file"
      sed -i.bak "s/$OLD/$NEW/g" "$file" && rm "${file}.bak"
    fi
  done
done

echo "✅ Done. Renamed project to '$NEW'."

