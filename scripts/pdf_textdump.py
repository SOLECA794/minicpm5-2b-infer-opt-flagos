import sys
import importlib

pdf = sys.argv[1] if len(sys.argv) > 1 else "proposals/_smoketest.pdf"

fitz = None
try:
    fitz = importlib.import_module("fitz")
except Exception:
    pass

if fitz is not None:
    d = fitz.open(pdf)
    print("pages:", d.page_count)
    print("--- page 1 ---")
    print(d[0].get_text()[:1200])
    print("--- page 2 ---")
    if d.page_count > 1:
        print(d[1].get_text()[:800])
else:
    from pypdf import PdfReader

    r = PdfReader(pdf)
    print("pages:", len(r.pages))
    print("--- page 1 ---")
    print(r.pages[0].extract_text()[:1200])