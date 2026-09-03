# include tokenized comparison (uses tokenized_sizes["tiktoken"] from filtered_scripts.json)
# python3 plot_ipa_vs_text.py \
#   --text-dir text --ipa-dir ipa \
#   --filtered-json filtered_files.json \
#   --tok-method tiktoken

# # save everything to plots_out/
# python3 plot_ipa_vs_text.py \
#   --text-dir text --ipa-dir ipa \
#   --filtered-json filtered_scripts.json \
#   --tok-method tiktoken \
#   --save --outdir plots_out --csv

# only keep languages that have tiktoken sizes
# python3 plot_ipa_vs_text.py \
#   --text-dir text --ipa-dir ipa \
#   --filtered-json filtered_tiles.json \
#   --tok-method tiktoken \
#   --skip-missing-tok


python3 plot_ipa_vs_text.py --text-dir text --ipa-dir ipa --save --outdir plots_out --csv

