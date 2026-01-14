# export_to_arxiv.sh
#!/bin/bash
tectonic titleofreport.tex --keep-logs --keep-intermediates

zip icml_arxiv.zip titleofreport.tex titleofreport.bib icml2025.sty icml2025.bst algorithmic.sty algorithm.sty fancyhdr.sty
