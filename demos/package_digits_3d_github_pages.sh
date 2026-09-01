#!/usr/bin/env bash
# Build a self-contained directory suitable for a GitHub Pages branch/artifact.
set -euo pipefail

SITE_DIR="${SITE_DIR:-dist/digits-3d-site}"
PORT="${PORT:-8000}"
PUBLISH="${PUBLISH:-false}"

python3 analysis/package_3d_trajectory_site.py --output-dir "${SITE_DIR}"

if [ "${PUBLISH}" = true ]; then
  REMOTE_URL="${REMOTE_URL:-$(git remote get-url origin)}"
  rm -rf "${SITE_DIR}/.git"
  git -C "${SITE_DIR}" init -q
  git -C "${SITE_DIR}" add .
  git -C "${SITE_DIR}" -c user.name="${GIT_AUTHOR_NAME:-nanoGPT demo publisher}" \
    -c user.email="${GIT_AUTHOR_EMAIL:-noreply@example.com}" \
    commit -q -m "Deploy 3D token trajectory demo"
  git -C "${SITE_DIR}" remote add origin "${REMOTE_URL}"
  git -C "${SITE_DIR}" push --force origin HEAD:gh-pages
  echo "Published ${SITE_DIR} to ${REMOTE_URL} branch gh-pages"
  exit 0
fi

cat <<EOF
Static site ready at ${SITE_DIR}

Preview:
  python3 -m http.server "${PORT}" -d "${SITE_DIR}"
  http://localhost:${PORT}/

Publish directly to the origin gh-pages branch:
  PUBLISH=true bash demos/package_digits_3d_github_pages.sh

Override the destination with REMOTE_URL. Publishing force-updates gh-pages.
Alternatively upload "${SITE_DIR}" as a GitHub Pages Actions artifact.
EOF
