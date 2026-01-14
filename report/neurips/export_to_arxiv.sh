#!/bin/bash
# export_to_arxiv.sh

set -x

tectonic --keep-logs --keep-intermediates titleofreport.tex

zip titleofreport_arxiv.zip *.tex *.bib *.sty

