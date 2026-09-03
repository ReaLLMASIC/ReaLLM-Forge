# Simplified Hanzi radical-location multicontext demo

This folder demonstrates a reversible multicontext split for simplified Hanzi.
`input.txt` may be either the bundled one-character-per-line corner-case fixture or an ordinary UTF-8 text corpus (for example `#Title:` / `#Poem:` records). `get_dataset.sh` treats the file as a character stream and creates one aligned multicontext timestep per Unicode code point:

- `char`: the original simplified character, making the representation a 1:1 bijection.
- `whole`, `left`, `right`, `top`, `bottom`, `enclosure`, `inside`, `corner`, `overlay`, `other`: radical/location signals.

`∅` means “this simplified Hanzi has nothing in this category.” `⧆` means “this
input code point is not treated as simplified Hanzi,” and is emitted in every lane.

The decomposition table is intentionally small and transparent for tests. It can
be replaced with a full Unihan/IDS-derived table without changing the lane
contract or downstream training commands.

Run:

```bash
bash data/simplified_hanzi_mc/get_dataset.sh
```

Each lane then contains `char_simplified_hanzi_mc/{train.bin,val.bin,meta.pkl}`.
