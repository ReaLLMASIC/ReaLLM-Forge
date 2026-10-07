#!/usr/bin/env bash
# Replace Bio-dashboards/ with the newest release folder from Meapy011/Bio-dash.
# Used by .github/workflows/sync-bio-dash.yml; also runnable by hand from the repo root.
#   SOURCE_REPO  repo to copy from        (default Meapy011/Bio-dash)
#   SOURCE_REF   branch or tag to copy    (default main)
#   DEST_DIR     folder in this repo      (default Bio-dashboards)
set -euo pipefail
SOURCE_REPO="${SOURCE_REPO:-Meapy011/Bio-dash}"
SOURCE_REF="${SOURCE_REF:-main}"
DEST_DIR="${DEST_DIR:-Bio-dashboards}"

work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
git clone -q --depth 1 --branch "$SOURCE_REF" "https://github.com/$SOURCE_REPO.git" "$work/src"
sha=$(git -C "$work/src" rev-parse --short HEAD)

# newest release folder: V1_Dashboard, V2.0_Dashboard ... V2.8_Dashboard, V3.0_Dashboard -> highest version
latest=$(cd "$work/src" && ls -d V*_Dashboard 2>/dev/null | sed 's#/$##' | sort -V | tail -1)
[ -n "$latest" ] || { echo "No V*_Dashboard folder found in $SOURCE_REPO@$SOURCE_REF"; exit 1; }
version="${latest%_Dashboard}"

# mirror it: replace the folder wholesale (git only records what actually changed),
# so files that no longer exist in Bio-dash are removed too
rm -rf "$DEST_DIR"
mkdir -p "$DEST_DIR"
cp -a "$work/src/$latest/." "$DEST_DIR/"
cat > "$DEST_DIR/SYNCED_FROM.md" << MD
# Bio-dashboards

This folder is a copy of **[$SOURCE_REPO](https://github.com/$SOURCE_REPO)**, release **$version**
(\`$latest/\`, commit \`$sha\`). It's synced automatically by \`.github/workflows/sync-bio-dash.yml\`,
so edit Bio-dash itself, not this copy -- changes made here are overwritten on the next sync.
MD

# un-ignore what the host repo's .gitignore would hide (it ignores *.txt), so requirements.txt
# files are tracked like any other file -- by this script and by plain git commands alike
cat >> "$DEST_DIR/.gitignore" << 'IGN'

# added by sync-bio-dash.sh: keep these even though the parent repo ignores *.txt
!requirements.txt
!*.txt
IGN
# -f: the host repo's .gitignore (e.g. *.txt) must not drop files such as requirements.txt
git add -A -f -- "$DEST_DIR"
if git diff --cached --quiet -- "$DEST_DIR"; then
    echo "Bio-dashboards already matches $SOURCE_REPO $version ($sha) -- nothing to do."
    echo "changed=false" >> "${GITHUB_OUTPUT:-/dev/null}"
    exit 0
fi
echo "Changes:"; git diff --cached --stat -- "$DEST_DIR" | tail -3
echo "changed=true" >> "${GITHUB_OUTPUT:-/dev/null}"
echo "message=Sync Bio-dashboards from Bio-dash $version ($sha)" >> "${GITHUB_OUTPUT:-/dev/null}"
