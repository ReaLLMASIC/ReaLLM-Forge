# Hanzi Factor binary format, version 1

This document specifies the structural byte format emitted by
`BinaryCodec`.  Multi-byte fields are concatenated without alignment unless
explicitly stated otherwise.

## Required profile

A decoder must have the same ordered component dictionary as the encoder.  A
dictionary is derived by recursively expanding every definition, grouping
identical expanded trees, and sorting the unique canonical prefix IDS strings
by Unicode scalar order.  The ordinal width is:

```text
w = ceil(log2(N)) = (N - 1).bit_length()
```

where `N` is the number of unique entries.  Thus `w = 0` for a one-entry
dictionary.  An empty dictionary cannot decode a component reference.

The SHA-256 dictionary fingerprint hashes the domain separator
`hanzi-factor-components-v1\0`, followed for every ordinal by a four-byte
big-endian UTF-8 byte length and the expanded IDS bytes.

A generic `BinaryCodec` uses that dictionary digest as its frame-profile
fingerprint. `HanziCodec` instead hashes the dictionary digest together with a
deterministic reverse-index digest binding every label to its expanded tree.
The resulting profile value is constant for the whole catalogue; it is not a
per-character ID.

The reverse-index digest is SHA-256 over the domain separator
`hanzi-factor-reverse-index-v1\0`, followed by labels in Unicode-scalar sort
order. Each entry is an eight-byte big-endian label-byte length, the UTF-8
label, an eight-byte big-endian expanded-IDS byte length, and the UTF-8
canonical IDS. The final HanziCodec profile is:

```text
SHA-256(
  "hanzi-factor-codec-profile-v1\0"
  || 32-byte component digest
  || 32-byte reverse-index digest
)
```

## Tree payload

Bits are written most-significant first.  The payload is a preorder tree.
Unicode-defined operator arity makes it self-delimiting.

| Prefix | Meaning | Following bits |
|---|---|---|
| `00` | binary operator | 4-bit operator index, then two nodes |
| `01` | ternary operator | 1-bit operator index, then three nodes |
| `10` | unary operator | 1-bit operator index, then one node |
| `110` | component reference | `w`-bit unsigned ordinal |
| `111` | raw scalar leaf | Elias gamma of `codepoint + 1` |

Binary operator indexes are fixed in this order:

```text
0 ⿰   1 ⿱   2 ⿴   3 ⿵   4 ⿶   5 ⿷   6 ⿸
7 ⿹   8 ⿺   9 ⿻  10 ⿼  11 ⿽  12 ㇯
```

Indexes 13–15 are invalid.  Ternary index 0 is `⿲` and index 1 is `⿳`.
Unary index 0 is `⿾` and index 1 is `⿿`.

The encoder compares the bit cost of a structural subtree with an available
component reference and selects the shorter representation.  A tie remains
structural.  By default the complete root may not be replaced by a component
reference; this prevents the dictionary ordinal from becoming a disguised
Hanzi ID.

## Frame

The default framed encoding is:

| Field | Size | Value |
|---|---:|---|
| magic | 2 bytes | ASCII `HF` |
| version | 1 byte | `1` |
| fingerprint length | 1 byte | decoder-profile value, default `8` |
| payload bit length | variable | canonical unsigned LEB128 |
| profile fingerprint | declared bytes | SHA-256 prefix |
| payload | `ceil(bits/8)` bytes | tree bits, then zero padding |

The decoder requires the incoming fingerprint length to equal its configured
profile length; a sender cannot downgrade this check.  It rejects a profile
mismatch, overlong LEB128, unknown operator/ordinal, invalid Unicode scalar,
truncation, extra bytes/bits, and non-zero byte padding.

Unframed values contain only the payload bytes.  Because the exact bit length
is absent, the decoder accepts at most seven final zero bits needed for byte
alignment.  The component dictionary remains required whenever the payload
uses a reference.

## Canonical identity

The byte stream decodes to an expanded structural tree, not directly to a
character.  A separate reverse index maps each expanded tree to all registered
labels.  A lookup succeeds only when exactly one label has that tree.  Zero
matches are unknown; two or more are an information-theoretic collision and
must not be guessed.
