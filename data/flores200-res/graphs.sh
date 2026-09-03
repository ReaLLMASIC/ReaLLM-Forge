#!/bin/bash
python3 plot_langscript_sizes_grouped.py --group-by script --color-by script --out by_script.png
python3 plot_langscript_sizes_grouped.py --group-by script --color-by region --out by_region_script.png
python3 plot_langscript_sizes_grouped.py --group-by region --color-by region --out by_region.png
python3 plot_langscript_sizes_grouped.py --group-by family --color-by family --out by_family.png
python3 plot_langscript_sizes_grouped.py --group-by family --color-by script --out by_family_script.png



