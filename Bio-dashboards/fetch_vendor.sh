#!/usr/bin/env bash
# Pulls the dashboards' front-end libraries into each static/vendor/ so the
# pages work with no internet (e.g. an offline Orin). Safe to re-run.
#
#   ./fetch_vendor.sh             download JS libs (wget) + verify checksums
#   ./fetch_vendor.sh --tailwind  also rebuild static/css/tailwind.css (needs Node/npx)
#
# The repo already ships these files; run this to re-download them or after
# bumping a version below (then update or clear its checksum).
set -euo pipefail
cd "$(dirname "$0")"

CDN="${CDN:-https://cdn.jsdelivr.net/npm}"   # override to use a mirror
# url-path|filename|sha256
LIBS=(
  "chart.js@3.9.1/dist/chart.min.js|chart.min.js|fbc45926e6b46845a0f905552a0e0b1331049bff1115ecf94dbe0904d895e710"
  "luxon@3.0.1/build/global/luxon.min.js|luxon.min.js|b90d11adca8043c371ccda991f5d5e5f1dea14b020c0481240dd55d0dd7e5253"
  "chartjs-adapter-luxon@1.2.0/dist/chartjs-adapter-luxon.min.js|chartjs-adapter-luxon.min.js|32bebde2ad9a4f051c0cbba522d3d3ad08c83230e25be2dec5009d153fe2b9b6"
  "chartjs-plugin-streaming@2.0.0/dist/chartjs-plugin-streaming.min.js|chartjs-plugin-streaming.min.js|4e591702ead92f73f79ee126598cabcc65757213f6a4945b99ac14ba8d227563"
)
CHART_DASHBOARDS=(polar_dashboard atmos_dashboard viatom_dashboard Omni-dashboard hydro_dashboard analysis_dashboard)
TAILWIND_DASHBOARDS=(polar_dashboard atmos_dashboard viatom_dashboard Omni-dashboard hydro_dashboard)

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

echo "📦 Downloading chart libraries..."
for entry in "${LIBS[@]}"; do
  IFS='|' read -r path file sum <<< "$entry"
  wget -q --show-progress -O "$tmp/$file" "$CDN/$path"
  if [[ -n "$sum" ]]; then
    echo "$sum  $tmp/$file" | sha256sum -c --quiet - \
      || { echo "❌ checksum mismatch for $file -- not installing"; exit 1; }
  fi
done

for d in "${CHART_DASHBOARDS[@]}"; do
  mkdir -p "$d/static/vendor"
  cp "$tmp"/*.js "$d/static/vendor/"
  echo "  ✅ $d/static/vendor"
done

if [[ "${1:-}" == "--tailwind" ]]; then
  command -v npx >/dev/null || { echo "❌ npx not found (install Node.js) -- skipping Tailwind"; exit 1; }
  echo "🎨 Rebuilding Tailwind CSS..."
  printf '@tailwind base;\n@tailwind components;\n@tailwind utilities;\n' > "$tmp/in.css"
  for d in "${TAILWIND_DASHBOARDS[@]}"; do
    npx -y tailwindcss@3 -i "$tmp/in.css" -o "$d/static/css/tailwind.css" \
      --content "$d/templates/*.html,$d/static/js/*.js" --minify 2>/dev/null
    echo "  ✅ $d/static/css/tailwind.css"
  done
fi

echo "Done. Pages now load everything from /static -- no internet needed."
