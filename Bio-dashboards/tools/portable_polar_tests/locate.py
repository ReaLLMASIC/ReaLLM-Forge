"""Where the portable_polar page is: $PORTABLE_POLAR, else a portable_polar folder next to this Bio-dash clone
(or in ~/Forks, or ~). The tests live here in Bio-dash; the page stays lean."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def page_dir():
    cands = [os.environ.get("PORTABLE_POLAR")]
    d = HERE
    for _ in range(6):
        d = os.path.dirname(d)
        cands.append(os.path.join(d, "portable_polar"))
    cands += [os.path.expanduser("~/Forks/portable_polar"), os.path.expanduser("~/portable_polar")]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "h10.js")):
            return os.path.abspath(c)
    raise SystemExit("Can't find portable_polar: set PORTABLE_POLAR=/path/to/portable_polar")
