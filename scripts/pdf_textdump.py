"""Dump the text of every page in a PDF, so rendered output can be verified
without opening a viewer.

Usage:
    python scripts/pdf_textdump.py <file.pdf>            # all pages
    python scripts/pdf_textdump.py <file.pdf> 5          # from page 5
    python scripts/pdf_textdump.py <file.pdf> 5 7        # pages 5..7
"""

from __future__ import annotations

import sys

import fitz  # PyMuPDF


def main(argv):
    if len(argv) < 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2

    path = argv[1]
    doc = fitz.open(path)
    first = int(argv[2]) if len(argv) > 2 else 1
    last = int(argv[3]) if len(argv) > 3 else doc.page_count

    print("%s: %d page(s)\n" % (path, doc.page_count))
    for i in range(first - 1, min(last, doc.page_count)):
        print("===== page %d =====" % (i + 1))
        print(doc[i].get_text().strip())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
