#!/usr/bin/env bash
# roomba-bootstrap.sh -- sets up a Jetson Orin's environment to run the
# Roomba controller in this same folder. Lives inside roomba/ itself
# (not standalone/top-level), so it assumes the repo is already checked
# out -- it just gets the environment ready: Python deps, the
# OpenCV-needs-GStreamer requirement, and serial port permissions.
#
# Some checks here can only be reported on, not auto-fixed (OpenCV's
# GStreamer support in particular) -- see the summary at the end.
#
# Usage:
#   ./roomba-bootstrap.sh

set -uo pipefail
# NOTE: deliberately not using -e here -- several checks below are meant
# to report a problem and continue (e.g. OpenCV/GStreamer, camera
# presence), not abort the whole script the way a missing build tool
# should for the LLM sweep bootstrap. Each step handles its own errors.

REAL_USER="${SUDO_USER:-${USER:-$(id -un)}}"

# This script lives inside roomba/ itself, so its own directory IS the
# roomba dir -- no repo URL/clone needed, unlike the LLM sweep harness
# bootstrap (which has no other canonical source for its files).
ROOMBA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOMBA_DIR"

ISSUES=()  # collected at the end into one clear summary

echo "roomba dir: $ROOMBA_DIR"
echo

# =============================================================================
echo "=== 1/4: system prerequisites ==="
# Unlike every other check in this script, a missing python3/pip3 is a
# hard stop, not a "report and continue" issue -- every step after this
# one runs python3 commands, so there's nothing meaningful left to check
# if this fails.
MISSING_SYS_PKGS=()
command -v python3 >/dev/null 2>&1 || MISSING_SYS_PKGS+=("python3")
command -v pip3 >/dev/null 2>&1 || MISSING_SYS_PKGS+=("python3-pip")

if [[ ${#MISSING_SYS_PKGS[@]} -eq 0 ]]; then
    echo "  found: python3, pip3"
else
    echo "  missing: ${MISSING_SYS_PKGS[*]} -- installing via apt"
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "  ERROR: apt-get not found -- this script only automates installs on" >&2
        echo "  Debian-based systems (which Raspberry Pi OS / Jetson L4T are). Install" >&2
        echo "  these manually and re-run: ${MISSING_SYS_PKGS[*]}" >&2
        exit 1
    fi
    SUDO_CMD=""
    if [[ "$(id -u)" -ne 0 ]]; then
        if ! command -v sudo >/dev/null 2>&1; then
            echo "  ERROR: not running as root and no sudo available -- install manually:" >&2
            echo "    apt-get update && apt-get install -y ${MISSING_SYS_PKGS[*]}" >&2
            exit 1
        fi
        SUDO_CMD="sudo"
    fi
    $SUDO_CMD apt-get update || echo "  WARNING: apt-get update had errors (possibly an unrelated broken repo) -- continuing anyway" >&2
    if ! $SUDO_CMD apt-get install -y "${MISSING_SYS_PKGS[@]}"; then
        echo "  ERROR: install failed -- can't continue without python3/pip3" >&2
        exit 1
    fi
    # Don't just trust apt-get's exit code -- verify the actual commands
    # are reachable now, same pattern as checking llama-bench exists after
    # building it in the LLM sweep bootstraps. A clean apt-get exit doesn't
    # always mean the binary landed somewhere on PATH.
    STILL_MISSING=()
    command -v python3 >/dev/null 2>&1 || STILL_MISSING+=("python3")
    command -v pip3 >/dev/null 2>&1 || STILL_MISSING+=("pip3")
    if [[ ${#STILL_MISSING[@]} -gt 0 ]]; then
        echo "  ERROR: apt-get reported success but still not on PATH: ${STILL_MISSING[*]}" >&2
        exit 1
    fi
    echo "  installed: ${MISSING_SYS_PKGS[*]}"
fi
echo

# =============================================================================
echo "=== 2/4: Python dependencies ==="
# flask, numpy, pyserial -- straightforward pip installs. NOT opencv here,
# see step 3: pip opencv is actively the wrong thing to install for this
# project, not just unnecessary.
MISSING_PY_PKGS=()
python3 -c "import flask" 2>/dev/null || MISSING_PY_PKGS+=("flask")
python3 -c "import numpy" 2>/dev/null || MISSING_PY_PKGS+=("numpy")
python3 -c "import serial" 2>/dev/null || MISSING_PY_PKGS+=("pyserial")  # pip name != import name

if [[ ${#MISSING_PY_PKGS[@]} -eq 0 ]]; then
    echo "  found: flask, numpy, pyserial"
else
    echo "  missing: ${MISSING_PY_PKGS[*]} -- installing"
    if ! pip install "${MISSING_PY_PKGS[@]}" --break-system-packages; then
        echo "  ERROR: pip install failed" >&2
        ISSUES+=("pip install failed for: ${MISSING_PY_PKGS[*]}")
    fi
fi
echo

# =============================================================================
echo "=== 3/4: OpenCV + GStreamer (camera pipeline requirement) ==="
# The control scripts use nvarguscamerasrc via cv2.VideoCapture(..., cv2.CAP_GSTREAMER),
# which needs an OpenCV build with GStreamer support compiled in. The
# NVIDIA-provided system OpenCV (from apt) has this; a plain
# `pip install opencv-python` does NOT and will shadow the correct one.
PIP_OPENCV_FOUND=()
for pkg in opencv-python opencv-python-headless opencv-contrib-python; do
    if pip show "$pkg" >/dev/null 2>&1; then
        PIP_OPENCV_FOUND+=("$pkg")
    fi
done
if [[ ${#PIP_OPENCV_FOUND[@]} -gt 0 ]]; then
    echo "  WARNING: found pip-installed OpenCV package(s): ${PIP_OPENCV_FOUND[*]}"
    echo "    These commonly shadow the correct system OpenCV and break the"
    echo "    camera pipeline. Recommended fix:"
    echo "      pip uninstall ${PIP_OPENCV_FOUND[*]} --break-system-packages"
    ISSUES+=("pip OpenCV package(s) present, may shadow system OpenCV: ${PIP_OPENCV_FOUND[*]}")
fi

if ! python3 -c "import cv2" 2>/dev/null; then
    echo "  cv2 not importable at all -- installing system package (python3-opencv)"
    SUDO_CMD=""
    if [[ "$(id -u)" -ne 0 ]]; then
        command -v sudo >/dev/null 2>&1 && SUDO_CMD="sudo"
    fi
    if command -v apt-get >/dev/null 2>&1; then
        $SUDO_CMD apt-get update || true
        $SUDO_CMD apt-get install -y python3-opencv
    else
        echo "  ERROR: apt-get not found -- install an OpenCV build with GStreamer support manually" >&2
        ISSUES+=("no apt-get available, cv2 not installed")
    fi
fi

GSTREAMER_OK=0
if python3 -c "import cv2" 2>/dev/null; then
    if python3 -c "
import cv2
info = cv2.getBuildInformation()
import sys
sys.exit(0 if ('GStreamer' in info and 'YES' in info.split('GStreamer')[1].split(chr(10))[0]) else 1)
" 2>/dev/null; then
        GSTREAMER_OK=1
        echo "  found: cv2 importable, GStreamer support confirmed"
    else
        echo "  WARNING: cv2 is importable but does NOT have GStreamer support."
        echo "    The camera pipeline will not work as-is. This script can't safely"
        echo "    auto-fix this (the correct fix depends on how OpenCV ended up"
        echo "    broken on this system) -- likely fixes, in order of how common:"
        echo "      1. Uninstall any pip OpenCV package (see above) and re-check"
        echo "      2. Reinstall the system package: sudo apt-get install --reinstall python3-opencv"
        echo "      3. Confirm you're not inside a venv that hides system site-packages"
        echo "         (venvs need --system-site-packages to see the NVIDIA OpenCV build)"
        ISSUES+=("cv2 importable but missing GStreamer support")
    fi
else
    echo "  WARNING: cv2 still not importable after install attempt"
    ISSUES+=("cv2 not importable")
fi
echo

# =============================================================================
echo "=== 4/4: serial port access ==="
if id -nG "$REAL_USER" 2>/dev/null | grep -qw dialout; then
    echo "  found: $REAL_USER already in dialout group"
else
    echo "  $REAL_USER not in dialout group -- adding (needed to open /dev/ttyUSB0 without sudo)"
    SUDO_CMD=""
    if [[ "$(id -u)" -ne 0 ]]; then
        command -v sudo >/dev/null 2>&1 && SUDO_CMD="sudo"
    fi
    if $SUDO_CMD usermod -aG dialout "$REAL_USER"; then
        echo "  added -- IMPORTANT: log out and back in (or run 'newgrp dialout')"
        echo "  for this to take effect in your current shell"
        ISSUES+=("dialout group just added -- needs fresh login to take effect")
    else
        echo "  ERROR: could not add $REAL_USER to dialout group" >&2
        ISSUES+=("failed to add $REAL_USER to dialout group")
    fi
fi

if ls /dev/ttyUSB* >/dev/null 2>&1; then
    echo "  found serial device(s): $(ls /dev/ttyUSB* 2>/dev/null | tr '\n' ' ')"
else
    echo "  no /dev/ttyUSB* device found right now -- fine if the CP2102/Roomba"
    echo "    isn't plugged in yet, just won't work until it is"
fi

if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet nvargus-daemon 2>/dev/null; then
    echo "  found: nvargus-daemon running (needed for the CSI camera pipeline)"
else
    echo "  NOTE: nvargus-daemon not confirmed running -- normal on non-Jetson"
    echo "    systems, but the CSI camera pipeline needs it on the Orin itself"
fi
echo

# =============================================================================
echo "=== Summary ==="
if [[ ${#ISSUES[@]} -eq 0 ]]; then
    echo "No issues found. Ready to run:"
    echo "  cd $ROOMBA_DIR"
    echo "  ./control-roomba"
else
    echo "${#ISSUES[@]} thing(s) worth checking before you rely on this:"
    for issue in "${ISSUES[@]}"; do
        echo "  - $issue"
    done
    echo
    echo "Once addressed:"
    echo "  cd $ROOMBA_DIR"
    echo "  ./control-roomba"
fi
