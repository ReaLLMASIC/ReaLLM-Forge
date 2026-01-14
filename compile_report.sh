#!/bin/bash
# compile_report.sh
# Example usage:
# ./compile_report.sh report_name dir1 file1 dir2 ...
# Output: report_name.tar.gz

report_name="$1"        # e.g. "report" -> "report.tar.gz"
shift 1                 # now $@ are additional files and dirs to include

echo "=== Compiling report: ${report_name}.tar.gz ==="

# Default directories and files (globs need expansion)
defaults=(./out ./logs ./csv_logs ./exploration_logs ./rem* ./*.csv ./*.yaml ./view_hp_log.py view_model_stats.py logging/ ./plot_view.py ./sample.txt ./explorations ./hp_searches)

# Expand defaults and collect existing items
dirs_and_files_to_include=()
for item in "${defaults[@]}"; do
    for match in $item; do
        if [ -e "$match" ]; then
            dirs_and_files_to_include+=("$match")
            echo "[OK] Adding: $match"
        else
            echo "[SKIP] Not found: $item"
        fi
    done
done

# Now check user-provided args
for x in "$@"; do
    if [ -e "$x" ]; then
        dirs_and_files_to_include+=("$x")
        echo "[OK] Adding: $x"
    else
        echo "[SKIP] Missing: $x"
    fi
done

# Final check
if [ ${#dirs_and_files_to_include[@]} -eq 0 ]; then
    echo "No valid files or directories found. Archive not created."
    exit 1
fi

# Create archive with transform (wrapper dir)
tar -czf "${report_name}.tar.gz" \
    --transform "s|^|${report_name}/|" \
    "${dirs_and_files_to_include[@]}"

echo "=== Done ==="
echo "Archive created: ${report_name}.tar.gz"
echo "To inspect: tar -tzf ${report_name}.tar.gz"

