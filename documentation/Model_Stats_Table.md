# Model Stats Table

The training script can report statistics for every tensor in the model.
Values include standard deviation, kurtosis, maximum, minimum and absolute
maximum for both weights and activations.

## Saving Statistics During Training

Add the `--print_model_stats_table <CSV_PATH>` argument when running
`train.py`. At each statistics evaluation step a table is printed and the raw
numbers are written to the specified CSV file.

```bash
python3 train.py --print_model_stats_table my_stats.csv
```

## Viewing Tables and Comparing Runs

Use `view_model_stats.py` to render a CSV as a colourised table or to compare
two CSV files. Deltas and percentages are shown with green for improvements and
red for regressions.

```bash
# Display a single table
python3 view_model_stats.py my_stats.csv

# Compare two tables and focus on stdev and abs_max columns
python3 view_model_stats.py run1.csv run2.csv --stats stdev,abs_max
```

Colour scales are computed per column. NaN values are highlighted in orange.
For kurtosis the gradient uses a logarithmic scale so that wide ranges remain
readable.

See the demo script [demos/adam_vs_adamw.sh](../demos/adam_vs_adamw.sh) for a
complete example that trains two models and compares their statistics.
