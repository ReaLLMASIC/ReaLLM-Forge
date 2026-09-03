# Simplified Hanzi radical-location multicontext demo

This folder demonstrates a reversible multicontext split for simplified Hanzi.
`input.txt` may be either the bundled one-character-per-line corner-case fixture or an ordinary UTF-8 text corpus (for example `#Title:` / `#Poem:` records). `get_dataset.sh` treats the file as a character stream and creates one aligned multicontext timestep per Unicode code point:

- `char`: the original simplified character, or `⧆` when the timestep is not simplified Hanzi.
- `non_hanzi`: the original non-simplified-Hanzi code point, or `∅` when the timestep is simplified Hanzi. Together with `char`, this makes the representation a full-text 1:1 bijection.
- `whole`, `left`, `right`, `top`, `bottom`, `enclosure`, `inside`, `corner`, `overlay`, `other`: radical/location signals.

`∅` means “this simplified Hanzi has nothing in this category” (and is also the empty value in `non_hanzi` for simplified Hanzi). `⧆` means “this
input code point is not treated as simplified Hanzi” in the `char`/radical lanes; the original code point is preserved in `non_hanzi` using line-safe escapes for control characters such as newlines.

The decomposition table is intentionally small and transparent for tests. It can
be replaced with a full Unihan/IDS-derived table without changing the lane
contract or downstream training commands.

Run:

```bash
bash data/simplified_hanzi_mc/get_dataset.sh
```

Each lane then contains `char_simplified_hanzi_mc/{train.bin,val.bin,meta.pkl}`.
