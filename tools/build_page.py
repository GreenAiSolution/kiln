"""
Wrap web/page.html into a standalone document for GitHub Pages.

web/page.html is the single source of truth for the visual writeup. It is a
fragment - no doctype, no head - because that is what the Artifact publisher
expects. GitHub Pages needs a complete document, so this script wraps the
same fragment rather than keeping a second copy that would drift.

    python3 tools/build_page.py
"""

import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "web", "page.html")
OUT = os.path.join(HERE, "docs", "index.html")

FAVICON = ("data:image/svg+xml,"
           "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E"
           "%3Ctext y='.9em' font-size='90'%3E%F0%9F%94%A5%3C/text%3E%3C/svg%3E")

DESC = ("A tensor compiler that emits its own ARM64 machine code. "
        "Pure Python standard library, no LLVM, no assembler, no numpy. "
        "Every instruction verified against Apple's own assembler.")


def main():
    with open(SRC) as f:
        frag = f.read()

    m = re.search(r"<title>(.*?)</title>", frag)
    title = m.group(1) if m else "KILN"
    frag = re.sub(r"<title>.*?</title>\s*", "", frag, count=1)

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{DESC}">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{DESC}">
<meta property="og:type" content="website">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="{FAVICON}">
<style>*{{margin:0;padding:0}}</style>
</head>
<body>
{frag}
</body>
</html>
"""
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(doc)
    print(f"wrote {OUT}  ({len(doc):,} bytes, title {title!r})")


if __name__ == "__main__":
    main()
