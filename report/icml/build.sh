#!/bin/bash
set -e  # Exit on error

# First pass: generate .aux and .bbl
tectonic --keep-intermediates titleofreport.tex

# Run bibtex
bibtex titleofreport

# Re-compile to incorporate bibliography
tectonic --keep-intermediates titleofreport.tex
tectonic titleofreport.tex

echo "✅ Build complete: titleofreport.pdf"

