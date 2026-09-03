#!/bin/bash
# compile_report.sh
# Example usage:
# ./compile_report.sh report_name dir1 file1 dir2 ...
# ./compile_report.sh report_name dir1 dir2 --clean
# Output: report_name.tar.gz

report_name="$1"        # e.g. "report" -> "report.tar.gz"
shift 1                 # now $@ are additional files and dirs to include

cleanup_requested=false
user_args=()
for arg in "$@"; do
    case "$arg" in
        --clean|-c)
            cleanup_requested=true
            ;;
        *)
            user_args+=("$arg")
            ;;
    esac
done

echo "=== Compiling report: ${report_name}.tar.gz ==="

# Default directories and files (globs need expansion)
defaults=(./out ./logs ./csv_logs ./exploration_logs ./rem* ./*.csv ./*.yaml ./view_hp_log.py view_model_stats.py logging/ ./plot_view.py ./sample.txt ./explorations ./hp_searches ./utils)

# Expand defaults and collect existing items
dirs_and_files_to_include=()
for item in "${defaults[@]}"; do
    # Rely on shell expansion for globs here
    for match in $item; do
        if [ -e "$match" ]; then
            dirs_and_files_to_include+=("$match")
            echo "[OK] Adding: $match"
        else
            # This handles cases where a glob didn't match anything and remains literal
            if [ "$match" == "$item" ]; then
                 echo "[SKIP] Not found: $item"
            fi
        fi
    done
done

# Now check user-provided args
for x in "${user_args[@]}"; do
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

if [ "$cleanup_requested" = true ]; then
    echo "=== Cleanup requested: removing archived items ==="
    
    # Check if we are in a git repo once, before the loop.
    # We also check if the 'git' command exists to avoid errors on minimal systems.
    in_git_repo=false
    if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        in_git_repo=true
    fi

    for item in "${dirs_and_files_to_include[@]}"; do
        if [ ! -e "$item" ]; then
            echo "[SKIP] Already removed or missing: $item"
            continue
        fi

        # Safety guardrails: Prevent deletion of current/parent dirs or root
        if [[ "$item" == "." || "$item" == ".." || "$item" == "/" ]]; then
             echo "[SKIP] Dangerous path detected, not deleting: $item"
             continue
        fi

        # Only check git tracking if we are definitely in a repo
        if [ "$in_git_repo" = true ] && git ls-files --error-unmatch "$item" >/dev/null 2>&1; then
            echo "[SKIP] Tracked by git, not deleting: $item"
            continue
        fi

        rm -rf "$item"
        echo "[DELETE] Removed: $item"
    done
fi
