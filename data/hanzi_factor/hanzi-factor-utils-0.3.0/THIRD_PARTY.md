# Third-party data

`hanzi-factor` contains no production Han decomposition catalogue.  This is
deliberate: a structural codec must pin and identify the exact catalogue used
to construct its dictionary.

The optional `scripts/fetch_ccd.py` helper downloads version 0.1.0 of the npm
package [`chinese-characters-decomposition`](https://www.npmjs.com/package/chinese-characters-decomposition),
whose package metadata declares the MIT license.  That snapshot packages the
Wikimedia Commons *Chinese characters decomposition* table.  The source table
describes itself as a purely graphical—not etymological—decomposition dataset.
It also contains uncertain and primitive/self rows; the loader and coverage
report preserve those distinctions instead of treating every row as a verified
recursive factorization. CCD enclosure rows do not state which Unicode surround
operator applies, so the strict adapter rejects them rather than guessing a
location. It likewise rejects the undocumented `*` topology.

The external JSON is not redistributed in this repository.  Review its source,
license, and data quality for your application before use.

## Optional OpenCC normalization dependency

The `normalize` extra installs `opencc-python-reimplemented` 0.1.7, distributed
under the Apache License 2.0. It is not bundled in this source tree or wheel.
Its OpenCC dictionaries and conversion profiles provide phrase-aware
Simplified/Traditional and regional normalization.
