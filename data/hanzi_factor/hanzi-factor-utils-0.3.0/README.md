# hanzi-factor

`hanzi-factor` is a dependency-free Python reference implementation for
turning a Han character's **canonical graphical decomposition** into an
ordered component tree, packing that tree into a compact deterministic bit
stream, and resolving the decoded tree back to a character when—and only
when—the catalogue makes that reverse mapping unique.

It implements the complete Unicode 17 IDS grammar: the 16 Ideographic
Description Characters at U+2FF0–U+2FFF plus the binary subtraction operator
U+31EF.  It also provides strict dataset loaders, recursive expansion with
cycle detection, collision analysis, UTF-range audits, and JSON/CSV/text
coverage reports.

## What “reversible” means

The encoded root contains structure, ordered component identities, and layout
operators.  It does not contain the root character's Unicode scalar or a row
number.  Decoding first recreates the canonical component tree; a reverse
index then returns the character whose expanded tree is exactly that tree.

There are two unavoidable qualifications:

1. If two characters have the same canonical tree, structure alone cannot
   distinguish them.  Formally, if `F(a) = F(b)` with `a != b`, no inverse of
   `F` can return both.  The library reports such collisions and refuses the
   ambiguous lookup instead of hiding a character ID in the payload.
2. A graphical primitive must remain an atomic component unless a lower-level
   stroke/path definition is supplied.  Encoding the identity of a true leaf
   is necessary information; calling all such leaves “radicals” does not make
   the 214 Kangxi indexing radicals a complete graphical alphabet.

This is therefore a testable structural bijection **relative to a pinned,
collision-free canonical catalogue**.  Exact font outlines, regional glyph
forms, proportions, and component deformations require a separate versioned
geometry profile.  Unicode explicitly notes that IDS rendering is not required
to recreate the described glyph.

## Quick start

Python 3.10 or newer is required.

```bash
python -m pip install -e .
python examples/roundtrip.py
python -m unittest discover -s tests -v
```

The example is self-contained and demonstrates `汉`, `国`, `语`, `清`, and
`森`, including tree expansion, binary encoding, decoding, and reverse lookup.

For a complete fresh-environment walkthrough—including OpenCC installation,
catalogue download, Simplified and Traditional normalization, document-to-IDS
conversion, binary round trips, and tests—run:

```bash
bash demos/setup_and_demo.sh
```

See [`demos/SETUP_AND_DEMO.md`](demos/SETUP_AND_DEMO.md) and the included mixed
Chinese input [`demos/sample_chinese.txt`](demos/sample_chinese.txt).

## Replace a document with prefix IDS

`scripts/text_to_ids.py` scans a UTF-8 document and replaces every character
covered by the selected decomposition catalogue. The default `expanded` form
recursively expands registered components; `direct` emits the catalogue's
canonical top-level IDS instead.

```bash
# stdin -> stdout, using the small test catalogue
printf '汉语，清!\n' | python scripts/text_to_ids.py \
  --ids tests/fixtures/sample_ids.txt --quiet
# ⿰氵又⿰讠⿱五口，⿰氵⿱龶月!

# A real document and the fetched CCD catalogue
python scripts/text_to_ids.py article.txt \
  --ccd data/ccd.json \
  --format expanded \
  --on-uncovered keep \
  --report article.ids.report.json \
  -o article.ids.txt
```

Non-Han text, whitespace, and punctuation are preserved. For Han characters
absent from the catalogue, `--on-uncovered` can `keep` them (the default), stop
with `error`, or emit a visible `<U+XXXX>` escape. `--wrap` renders every
replacement as `⟦IDS⟧`, which is useful while inspecting mixed text. Use
`--in-place` to atomically replace a named input file.

The expanded form is prefix notation and operator arity makes each IDS tree
self-delimiting. An operand absent from the catalogue remains an atomic leaf;
use the coverage audit's `--no-leaf-fallback` mode when the workflow must first
prove complete recursive closure.

The same operation is available as a Python API:

```python
from hanzi_factor import factorize_text

mapping = {"汉": "⿰氵又", "清": "⿰氵青", "青": "⿱龶月"}
result = factorize_text("汉，清", mapping, expanded=True)
print(result.text)  # ⿰氵又，⿰氵⿱龶月
```

## Restore a prefix-IDS document

Use the same pinned catalogue to invert a document produced by
`text_to_ids.py`:

```bash
python scripts/ids_to_text.py article.ids.txt \
  --ccd data/ccd.json \
  --report article.restored.report.json \
  -o article.restored.txt
```

The parser consumes an IDS operator and exactly the operands required by its
arity, then resumes copying ordinary punctuation, numbers, whitespace, emoji,
and Latin script. It also decodes `<U+XXXX>` fallbacks and recognizes the
optional `⟦IDS⟧` form automatically. Both expanded and direct IDS reverse
through the catalogue's canonical expanded index.

Unknown trees and structurally ambiguous trees are errors by default. This is
the only safe default: if two registered characters have the same expanded
graphical tree, IDS alone contains no bit that can recover which identity was
original. `--on-ambiguous first` is available only when deterministic lossy
selection is acceptable; `--on-unknown keep` preserves unknown expressions.

For an exact document round trip, use `--on-uncovered escape` in the forward
command. If an uncovered Han scalar is merely kept, a bare one-node Han operand
can be indistinguishable from an intentionally emitted atomic IDS root. The
escape preserves that distinction while remaining readable and reversible.

Python API:

```python
from hanzi_factor import restore_text

restored = restore_text("⿰氵又，⿰氵⿱龶月", mapping)
print(restored.text)  # 汉，清
```

## Normalize Simplified or Traditional Chinese

Simplified/Traditional normalization is phrase-sensitive and is therefore a
separate OpenCC-backed operation rather than a character-by-character table.
Install the optional dependency once:

```bash
python -m pip install '.[normalize]'
```

Then select a target and, optionally, a regional profile:

```bash
# Generic Traditional Chinese
python scripts/normalize_chinese.py input.txt \
  --to traditional -o traditional.txt

# Taiwan orthography and phrase vocabulary (软件 -> 軟體)
python scripts/normalize_chinese.py input.txt \
  --to traditional --variant taiwan-phrases -o taiwan.txt

# Convert a Taiwan Traditional document to Simplified in place
python scripts/normalize_chinese.py input.txt \
  --to simplified --variant taiwan-phrases --in-place
```

Profiles are `generic`, `taiwan`, `taiwan-phrases`, and `hong-kong`. Conversion
preserves non-Chinese content, but it is not a bijection: several traditional
forms may collapse to one simplified form, and a later reverse conversion may
not reproduce the original wording. Keep the source document when identity or
editorial history matters.

## Audit a real dataset

Production decomposition data is deliberately external: the catalogue and its
version are part of the codec contract.  A pinned fetch helper is included for
the 21,169-record MIT-declared npm snapshot of the Wikimedia Commons graphical
decomposition table:

```bash
python scripts/fetch_ccd.py data/ccd.json
python scripts/coverage_utf.py \
  --ccd data/ccd.json \
  --range 4E00-9FFF \
  --no-leaf-fallback \
  --json coverage-cjk-basic.json
```

The audit walks every Unicode scalar in the inclusive range.  Its categories
distinguish:

- a valid structural IDS;
- an explicit primitive/self row;
- a definition that needs an unknown leaf fallback;
- an uncertain or rejected source row;
- a missing root;
- an expansion cycle;
- a canonical collision; and
- a failed binary/tree round trip.

Use `--fail-under` in CI to enforce a chosen recursive-coverage percentage. Run
`python scripts/coverage_utf.py --help` for range presets, multiple ranges,
plain IDS/TSV input, permissive leaf handling, and report formats.

### List uncovered characters

Extract failures from any JSON coverage report without rerunning the audit:

```bash
# Full no-root-ID requirement: recursive closure + unique inverse + binary pass
python scripts/list_uncovered.py coverage-cjk-basic.json \
  --output uncovered-bijective.tsv

# Only characters with no accepted top-level decomposition, one per line
python scripts/list_uncovered.py coverage-cjk-basic.json \
  --criterion direct \
  --format characters \
  --output uncovered-direct.txt
```

The default TSV includes the code point, character, audit status, failure
reason, and accepted IDS (when present). Other selectable thresholds are
`recursive`, `unique`, and `binary`; JSON output is also available.

### Reference audit of the pinned CCD snapshot

The included implementation was exercised over every scalar from U+4E00
through U+9FFF (20,992 targets), not just the small unit-test fixture:

| Measure | Result |
|---|---:|
| Strictly accepted root records | 19,697 (93.8310%) |
| Recursively closed with no missing-leaf fallback | 13,884 (66.1395%) |
| Unique structural inverses after collisions | 13,739 (65.4487%) |
| Strict expanded trees passing binary round trip | 13,884 / 13,884 |
| Strict reverse collisions | 71 groups / 145 characters |
| Missing/rejected roots | 1,295 |

In permissive component-leaf mode, all 19,697 accepted roots expand and pass
the binary round trip, but that is **not** full radical closure: absent
component definitions are then treated as atomic leaves.  The strict JSON and
text reports are in `artifacts/`; they retain every missing code point, source
diagnostic, collision group, and payload-bit measurement.

There is no contiguous “Simplified Hanzi” Unicode range: the CJK blocks unify
Chinese, Japanese, Korean, and Vietnamese usage and contain both simplified
and traditional forms.  For a claim specifically about simplified Chinese,
pass a repertoire file for the exact standard or product vocabulary you need;
the U+4E00–U+9FFF audit is a useful stress test, not a definition of simplified
Chinese.

## Data formats

The generic text loader accepts the common CJKVI-style form:

```text
U+6C49<TAB>汉<TAB>⿰氵又
U+8BED<TAB>语<TAB>⿰讠⿱五口
```

It also accepts a two-column `character<TAB>IDS` form. Blank lines and lines
beginning with `#` are ignored. Root keys must be exactly one Unicode scalar.
During the audit, every IDS is parsed structurally; malformed, trailing,
multi-root, and duplicate/conflicting records are surfaced rather than silently
repaired.

The CCD JSON adapter converts only losslessly placeable source layouts to
Unicode IDS operators. It rejects uncertainty markers, opaque concatenations,
`回` rows whose enclosure side is merely implicit, and the undocumented `*`
topology. These rejections are deliberate: guessing `⿴` for every enclosure
would encode wrong locations. See [THIRD_PARTY.md](THIRD_PARTY.md) before using
that data in a product.

## Canonical model

IDS is prefix notation, so operator arity makes parentheses unnecessary:

```text
语  = ⿰讠⿱五口
森  = ⿱木⿰木木
国  = ⿴囗玉
```

`parse_ids()` creates an immutable ordered tree.  `format_ids()` is its unique
whitespace-free serialization.  A component dictionary recursively substitutes
known leaf definitions until primitives remain, detects cycles, deduplicates
equal expanded components, and computes a stable fingerprint.  Reverse lookup
uses the fully expanded tree, making it independent of which reusable
subcomponents happened to be selected by the encoder.

The compact payload is a preorder tree code.  Operators use a small fixed
alphabet; known component subtrees use deterministic dictionary ordinals; and
unknown Unicode-scalar leaves use a bounded integer escape.  A framed payload
adds a magic/version marker, codec-profile fingerprint, and exact bit length.
For `BinaryCodec` that profile binds the component table; for `HanziCodec` it
binds both the component table and the complete label-to-tree reverse catalogue.
It is one global catalogue hash, not a per-character ID.  Unframed mode is
smaller when the caller already fixes the same profile out of band. Canonical
padding and integer encodings are checked, and depth/node limits apply to the
fully expanded result—including subtrees recovered through component refs.

The exact version-1 bit and frame grammar is documented in
[`docs/FORMAT.md`](docs/FORMAT.md).

Root component references are disabled by default.  Otherwise a dictionary
could encode `汉` as the ordinal for `汉`, producing a compact lookup ID rather
than a factorization.

## Why the audit is part of the implementation

A source file having a row for a character is not proof of full factorization.
A row may simply define the character as itself, contain `？`, depend on a
missing component, participate in a cycle, or collide after recursive
expansion.  Coverage is consequently measured at several increasingly strong
levels:

```text
row present -> valid tree -> recursively closed -> binary round trip
            -> unique reverse lookup -> optional exact geometry profile
```

Only the unique reverse-lookup stage establishes the no-root-ID structural
bijection.  The final geometry stage is required before claiming pixel- or
font-exact glyph recreation.

## References

- [Unicode 17.0, Chapter 18.2: Ideographic Description
  Characters](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-18/)
- [Unicode Standard Annex #38: Unicode Han
  Database](https://www.unicode.org/reports/tr38/)
- [Wikimedia Commons: Chinese characters
  decomposition](https://commons.wikimedia.org/wiki/Commons:Chinese_characters_decomposition)
- [CJKVI IDS data](https://github.com/cjkvi/cjkvi-ids)
- [Make Me a Hanzi](https://github.com/skishore/makemeahanzi) for a possible
  stroke-vector geometry layer (subject to its separate data licenses)

## License

The implementation is MIT licensed.  External catalogues retain their own
licenses and are not included.
