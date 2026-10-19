#!/usr/bin/env python3
"""Build the compute-resource application PDF for FlagOS S2 Track 2.

Inputs
------
  proposals/01-initial-technical-plan.md   the technical proposal body
  proposals/team.yaml                      team info (fill this in)

Output
------
  proposals/算力申请-<team_name>.pdf

The competition requires the file to be named exactly
``算力申请-[队名].pdf`` and asks for "初始技术方案 + 团队介绍" (60% / 40% of the
review weight).

Usage
-----
    python scripts/build_proposal.py --check     # validate team.yaml, then exit
    python scripts/build_proposal.py             # render the PDF
    python scripts/build_proposal.py --out x.pdf # custom output path

Requires: reportlab, PyYAML.  Fonts: Microsoft YaHei / SimHei / SimSun from the
Windows font directory, with the built-in Adobe CJK font as a last resort.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROPOSALS = ROOT / "proposals"
TECH_PLAN = PROPOSALS / "01-initial-technical-plan.md"
TEAM_YAML = PROPOSALS / "team.yaml"

# --------------------------------------------------------------------------
# Fonts
# --------------------------------------------------------------------------
FONT_DIR = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"

# A4 geometry in points, kept as module constants so the table column sizer can
# budget against the real text width instead of guessing.
from reportlab.lib.pagesizes import A4 as _A4
from reportlab.lib.units import mm as _mm

A4_WIDTH, A4_HEIGHT = _A4
LEFT_MARGIN = 18 * _mm
RIGHT_MARGIN = 16 * _mm

BODY_FONT, BOLD_FONT, MONO_FONT = "Body", "BodyBold", "Mono"


def register_fonts() -> None:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase.ttfonts import TTFont

    def try_ttf(name: str, *filenames: str) -> str | None:
        for fn in filenames:
            p = FONT_DIR / fn
            if p.exists():
                try:
                    pdfmetrics.registerFont(TTFont(name, str(p)))
                    FONT_FILES[name] = p
                    return name
                except Exception:
                    continue
        return None

    ok_body = try_ttf(BODY_FONT, "msyh.ttc", "simsun.ttc", "Deng.ttf")
    ok_bold = try_ttf(BOLD_FONT, "msyhbd.ttc", "simhei.ttf", "Deng.ttf")
    ok_mono = try_ttf(MONO_FONT, "consola.ttf", "cour.ttf")

    if ok_body is None:  # last resort: built-in Adobe CJK
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        ok_body = "STSong-Light"
    if ok_bold is None:
        ok_bold = ok_body
    if ok_mono is None:
        ok_mono = ok_body

    pdfmetrics.registerFontFamily(
        BODY_FONT,
        normal=ok_body,
        bold=ok_bold,
        italic=ok_body,
        boldItalic=ok_bold,
    )
    globals()["BODY_FONT"] = ok_body
    globals()["BOLD_FONT"] = ok_bold
    globals()["MONO_FONT"] = ok_mono
    FONT_FILES.setdefault(ok_body, FONT_DIR / "msyh.ttc")
    _CMAP_CACHE.clear()


# --------------------------------------------------------------------------
# Inline markdown -> reportlab mini-HTML
# --------------------------------------------------------------------------
def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_PLACEHOLDERS = []

# The CJK text fonts ship no emoji glyphs, so emoji would render as blank or
# tofu. Map the handful we actually use to their plain-text meaning.
_EMOJI_MAP = {
    "\u274c": "\u5426",        # cross mark      -> 否
    "\u2705": "\u662f",        # check mark      -> 是
    "\u26a0\ufe0f": "\u6ce8\u610f\uff1a",  # warning
    "\u26a0": "\u6ce8\u610f\uff1a",
    "\u2b50": "\u2605",        # star            -> black star (in CJK font)
    "\u2757": "\uff01",        # exclamation
    "\u23f3": "\u2026",        # hourglass       -> ellipsis
    "\u2b06\ufe0f": "\u2191",
    "\u2b07\ufe0f": "\u2193",
    "\u27a1\ufe0f": "\u2192",
    "\U0001f4b0": "",          # money bag
    "\u2714\ufe0f": "\u221a",
    "\u2714": "\u221a",
    # U+2713/U+2717 are missing from Microsoft YaHei (checked with fontTools),
    # so they render as tofu; U+221A / U+00D7 are present in every CJK font.
    "\u2713": "\u221a",
    "\u2717": "\u00d7",
}


def _demojify(text: str) -> str:
    for emo, repl in _EMOJI_MAP.items():
        text = text.replace(emo, repl)
    return text


def inline(text: str) -> str:
    """Convert inline markdown to the small HTML subset reportlab understands."""
    text = _demojify(text)
    # Protect inline code first so its content is not processed further.
    codes: list[str] = []

    def stash(m: re.Match[str]) -> str:
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = esc(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<link href="\2">\1</link>', text)

    def unstash(m: re.Match[str]) -> str:
        body = codes[int(m.group(1))]
        # Consolas carries no CJK glyphs; a span mixing them would lose the
        # Chinese characters entirely, so fall back to the body face.
        face = BODY_FONT if _has_cjk(body) else MONO_FONT
        return f'<font face="{face}" size="8.5">{esc(body)}</font>'

    return re.sub(r"\x00(\d+)\x00", unstash, text)


# --------------------------------------------------------------------------
# Block markdown -> platypus flowables
# --------------------------------------------------------------------------
def build_story(md: str, styles):
    from reportlab.lib import colors
    from reportlab.platypus import (
        HRFlowable,
        KeepTogether,
        ListFlowable,
        ListItem,
        PageBreak,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )

    story: list = []
    lines = md.split("\n")
    i = 0
    n = len(lines)

    def flush_list(items, ordered):
        if not items:
            return
        # Render bullets as plain paragraphs with bulletText rather than
        # ListFlowable: reportlab's bullet-type handling renders the wrong
        # marker for the default styles we build here.
        for idx, text in enumerate(items, 1):
            marker = f"{idx}." if ordered else "\u2022"
            story.append(
                Paragraph(inline(text), styles["bullet"], bulletText=marker)
            )
        story.append(Spacer(1, 4))

    while i < n:
        line = lines[i]

        # ---- fenced code block ----
        if line.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            code = "\n".join(buf).rstrip()
            if code:
                story.append(_code_block(code, styles))
            story.append(Spacer(1, 6))
            continue

        # ---- table ----
        if line.lstrip().startswith("|") and i + 1 < n and re.match(
            r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]
        ):
            header = _split_row(line)
            i += 2
            rows = []
            while i < n and lines[i].lstrip().startswith("|"):
                rows.append(_split_row(lines[i]))
                i += 1
            story.append(_table(header, rows, styles))
            story.append(Spacer(1, 8))
            continue

        # ---- heading ----
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            story.append(Paragraph(inline(m.group(2)), styles[f"h{level}"]))
            if level == 1:
                story.append(HRFlowable(width="100%", thickness=0.8,
                                        color=colors.HexColor("#333333"),
                                        spaceBefore=2, spaceAfter=8))
            i += 1
            continue

        # ---- horizontal rule ----
        if re.match(r"^\s*(---|\*\*\*)\s*$", line):
            story.append(HRFlowable(width="100%", thickness=0.5,
                                    color=colors.HexColor("#BBBBBB"),
                                    spaceBefore=6, spaceAfter=8))
            i += 1
            continue

        # ---- blockquote ----
        if line.lstrip().startswith(">"):
            buf = []
            while i < n and lines[i].lstrip().startswith(">"):
                buf.append(lines[i].lstrip()[1:].strip())
                i += 1
            story.append(Paragraph(inline(" ".join(buf)), styles["quote"]))
            story.append(Spacer(1, 6))
            continue

        # ---- list ----
        if _BULLET_RE.match(line) or _ORDERED_RE.match(line):
            ordered = bool(_ORDERED_RE.match(line))
            items: list[str] = []
            while i < n:
                cur = lines[i]
                mm = _ORDERED_RE.match(cur) if ordered else _BULLET_RE.match(cur)
                if mm:
                    items.append(mm.group(1))
                    i += 1
                    continue
                # Indented, non-blank lines continue the item they follow;
                # without this, a wrapped item is split into an item plus a
                # stray paragraph and the numbering restarts at 1.
                if (
                    items
                    and cur.strip()
                    and cur[:1] in (" ", "\t")
                    and not _ORDERED_RE.match(cur)
                    and not _BULLET_RE.match(cur)
                ):
                    items[-1] = items[-1] + " " + cur.strip()
                    i += 1
                    continue
                break
            flush_list(items, ordered)
            continue

        # ---- blank ----
        if not line.strip():
            i += 1
            continue

        # ---- paragraph (join wrapped lines) ----
        buf = [line]
        i += 1
        while i < n and lines[i].strip() and not re.match(
            r"^\s*(#{1,4}\s|[-*+]\s|\d+\.\s|\||>|```|---)", lines[i]
        ):
            buf.append(lines[i])
            i += 1
        story.append(Paragraph(inline(" ".join(x.strip() for x in buf)), styles["body"]))
        story.append(Spacer(1, 5))

    return story


def _split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def _is_wide(ch):
    return (
        "\u1100" <= ch <= "\u115f"
        or "\u2e80" <= ch <= "\ua4cf"
        or "\uac00" <= ch <= "\ud7a3"
        or "\uf900" <= ch <= "\ufaff"
        or "\ufe30" <= ch <= "\ufe6f"
        or "\uff00" <= ch <= "\uff60"
        or "\uffe0" <= ch <= "\uffe6"
    )


def _display_len(s):
    """Width in em-ish units; CJK glyphs are about twice as wide as latin."""
    return sum(2.0 if _is_wide(ch) else 1.0 for ch in s)


_BULLET_RE = re.compile(r"^\s*[-*+]\s+(.*)$")
_ORDERED_RE = re.compile(r"^\s*\d+\.\s+(.*)$")
_CELL_SPLIT = re.compile(r"[\s\u3000]+")


def _fits_one_line(para, width):
    return para.wrap(width, 1e6)[1] <= para.style.leading + 0.01


def _natural_width(para, cap):
    """Smallest width at which the whole cell stays on a single line."""
    lo = para.minWidth()
    if lo >= cap or _fits_one_line(para, cap):
        return cap
    hi = cap
    while hi - lo > 0.5:
        mid = (lo + hi) / 2.0
        if _fits_one_line(para, mid):
            hi = mid
        else:
            lo = mid
    return hi


def _col_widths(data, ncols):
    """Size columns from measured text instead of splitting the page evenly.

    An even split wraps short-but-wide cells such as ``[4096,1024,64,256]`` and
    header words such as ``TTFT`` onto two lines, which makes the baseline table
    hard to read. Each column first gets enough room for its widest unbreakable
    fragment (so nothing is split mid-token), then the leftover space is shared
    in proportion to how much more each column wants.

    Widths are measured on the rendered Paragraphs, not on the raw markdown:
    ``inline()`` rewrites backticked spans to a wider monospace font, so
    measuring the source text underestimates and the column then overflows.
    """
    # Frame padding is 6pt per side on top of the doc margins; ignoring it
    # shifts every table 6pt right and pushes it 6pt past the right margin.
    avail = A4_WIDTH - LEFT_MARGIN - RIGHT_MARGIN - 12.0
    pad = 8.0  # LEFTPADDING + RIGHTPADDING below

    need_min, need_max = [], []
    for c in range(ncols):
        mn, mx = 1.0, 1.0
        for row in data:
            if c >= len(row):
                continue
            para = row[c]
            mn = max(mn, para.minWidth())
            mx = max(mx, _natural_width(para, avail))
        need_min.append(mn + pad)
        need_max.append(mx + pad)

    if sum(need_min) >= avail:
        # Cannot honour every fragment; fall back to proportional minimums.
        scale = avail / sum(need_min)
        return [w * scale for w in need_min]

    widths = list(need_min)
    spare = avail - sum(need_min)
    extra_wanted = [max(0.0, need_max[i] - need_min[i]) for i in range(ncols)]
    total_extra = sum(extra_wanted)
    if total_extra <= 0:
        # Everything already fits on one line: distribute the slack evenly.
        return [w + spare / ncols for w in widths]
    for i in range(ncols):
        widths[i] += spare * extra_wanted[i] / total_extra
    return widths


def _table(header, rows, styles):
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    ncols = max([len(header)] + [len(r) for r in rows]) if rows or header else 1
    data = [[_cell_para(h, styles, bold=True) for h in _pad(header, ncols)]]
    for r in rows:
        data.append([_cell_para(c, styles) for c in _pad(r, ncols)])

    t = Table(data, repeatRows=1, hAlign="LEFT",
              colWidths=_col_widths(data, ncols))
    # Deliberately NOT wrapped in KeepTogether: reportlab's keepWithNext glue
    # skips any follower that _ktAllow() rejects, and it rejects _ContainerSpace
    # subclasses -- so wrapping a table would silently disable the heading glue.
    # Table headers repeat on continuation via repeatRows=1 above.
    t.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EFEFEF")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def _pad(row: list[str], n: int) -> list[str]:
    return (row + [""] * n)[:n]


def _cell_para(text: str, styles, bold: bool = False):
    from reportlab.platypus import Paragraph

    txt = inline(text) or "&nbsp;"
    return Paragraph(f"<b>{txt}</b>" if bold else txt, styles["cell"])


FONT_FILES: dict[str, "Path"] = {}
_CMAP_CACHE: dict[str, set[int]] = {}


# Closing punctuation that may not open a line. reportlab ships only U+3001 /
# U+3002 and a few brackets in ALL_CANNOT_START; the fullwidth comma, semicolon
# and friends are just as illegal and appear constantly in Chinese prose.
_CANNOT_START_EXTRA = "\uff0c\uff1b\uff1a\uff01\uff1f\uff3d\uff5d\u300b\u3009\u3015\u2026"


def _cannot_start_set() -> set[str]:
    from reportlab.lib.textsplit import ALL_CANNOT_START

    return set(ALL_CANNOT_START) | set(_CANNOT_START_EXTRA)


def _is_word_char(ch) -> bool:
    """Characters that must not be separated from their neighbours.

    Section signs and fullwidth slashes glue a number to its prefix, so
    ``\u00a72.3`` stays together instead of breaking after the sign.
    """
    if ch in "\u00a7\u00b6\uff0f/":
        return True
    return ch.isascii() and (ch.isalnum() or ch in "_.")


def _break_cjk(chars, widths, max_widths, cannot):
    """Break a run of characters into lines that respect width and kinsoku.

    Returns a list of ``(start, end)`` index pairs. Starting from the greedy
    break, two adjustments are applied while walking *backwards*:

    * a closing punctuation mark may not open the next line, and
    * a Latin identifier such as ``TRITON_ATTN`` is not split if it can be
      avoided within a short look-back.

    The break only ever moves earlier, so a line never exceeds ``max_width``.
    """
    lines = []
    start = 0
    n = len(chars)
    while start < n:
        max_w = max_widths[min(len(lines), len(max_widths) - 1)]
        w = 0.0
        end = start
        while end < n:
            cw = widths[end]
            if w + cw > max_w + 1e-6 and end > start:
                break
            w += cw
            end += 1
        if end >= n:
            lines.append((start, n))
            break

        # Closing punctuation may not open the next line.
        while end > start + 1 and end < n and chars[end] in cannot:
            end -= 1
        # Do not split an identifier across the line break.
        if end < n and _is_word_char(chars[end - 1]) and _is_word_char(chars[end]):
            limit = max(start + 1, end - 24)
            while end > limit and _is_word_char(chars[end - 1]):
                end -= 1

        lines.append((start, end))
        start = end
    return lines


def install_cjk_kinsoku() -> bool:
    """Make CJK line breaking respect width, kinsoku and identifier integrity.

    reportlab breaks CJK text two ways: a paragraph of a single font run goes
    through ``textsplit.wordSplit``, a paragraph carrying inline markup goes
    through ``paragraph.cjkFragSplit``. Both share two defects for Chinese
    prose: only a single character is pushed back, so ``\uff09\u3002`` leaves the
    full stop alone at the start of the next line, and the push-back happens
    after the fact, so it can split a Latin token. Route both through one
    correct breaker. Returns True when the patches took effect.
    """
    try:
        from reportlab.lib import textsplit as _ts
        from reportlab.platypus import paragraph as _para

        cannot = _cannot_start_set()

        # --- single-fragment paragraphs -----------------------------------
        def word_split(word, maxWidths, fontName, fontSize, encoding="utf8"):
            uword = word if _ts.isUnicode(word) else word.decode(encoding)
            widths = _ts.getCharWidths(uword, fontName, fontSize)
            if not isinstance(maxWidths, (list, tuple)):
                maxWidths = [maxWidths]

            out = []
            for start, end in _break_cjk(uword, widths, maxWidths, cannot):
                extra = maxWidths[min(len(out), len(maxWidths) - 1)] - sum(
                    widths[start:end]
                )
                text = uword[start:end]
                out.append([extra, text.strip() if start else text])
            if not _ts.isUnicode(word):
                out = [[e, t.encode(encoding)] for e, t in out]
            return out

        _ts.wordSplit = word_split
        _para.wordSplit = word_split

        # --- multi-fragment paragraphs ------------------------------------
        def cjk_frag_split(frags, maxWidths, calcBounds, encoding="utf8"):
            U = []
            for f in frags:
                text = f.text
                if _para.isBytes(text):
                    text = text.decode(encoding)
                if text:
                    U.extend([_para.cjkU(t, f, encoding) for t in text])
                else:
                    U.append(_para.cjkU(text, f, encoding))

            if not isinstance(maxWidths, (list, tuple)):
                maxWidths = [maxWidths]
            widths = [u.width for u in U]
            chars = [str(u) for u in U]

            # An explicit line break inside the markup ends a line early.
            bounds, last = [], 0
            for idx, u in enumerate(U):
                if hasattr(u.frag, "lineBreak"):
                    bounds.append((last, idx))
                    last = idx
            tail = _break_cjk(chars[last:], widths[last:], maxWidths, cannot)
            bounds.extend((last + s, last + e) for s, e in tail)

            lines = []
            for start, end in bounds:
                if end <= start:
                    continue
                max_w = maxWidths[min(len(lines), len(maxWidths) - 1)]
                used = sum(widths[start:end])
                lines.append(
                    _para.makeCJKParaLine(
                        U[start:end], max_w, used, max_w - used, False, calcBounds
                    )
                )
            if not lines:
                return _para.ParaLines(
                    kind=1, lines=[],
                    fontSize=frags[0].fontSize, fontName=frags[0].fontName,
                )
            return _para.ParaLines(kind=1, lines=lines)

        _para.cjkFragSplit = cjk_frag_split
        return True
    except Exception:
        return False


def _font_cmap(face: str) -> set[int]:
    """Codepoints the registered face can actually draw."""
    if face in _CMAP_CACHE:
        return _CMAP_CACHE[face]
    cps: set[int] = set()
    path = FONT_FILES.get(face)
    if path is not None and path.exists():
        try:
            from fontTools.ttLib import TTCollection, TTFont

            if path.suffix.lower() == ".ttc":
                fonts = TTCollection(str(path)).fonts
                ft = fonts[0]
            else:
                ft = TTFont(str(path))
            cps = set(ft.getBestCmap() or {})
        except Exception:
            cps = set()
    _CMAP_CACHE[face] = cps
    return cps


def _face_for(text: str, preferred: str) -> str:
    """Preferred face when it can draw every glyph, else the CJK body face.

    Monospace faces have no CJK coverage and miss a few symbols (Consolas has
    no U+2713), so text mixing them would silently lose characters.
    """
    cps = _font_cmap(preferred)
    if not cps:
        return preferred
    body = _font_cmap(BODY_FONT)
    if body and all(ord(ch) in body for ch in text):
        for ch in text:
            if ch not in " \t\r\n" and ord(ch) not in cps:
                return BODY_FONT
        return preferred
    return preferred


def _has_cjk(text: str) -> bool:
    return any(
        "\u4e00" <= ch <= "\u9fff" or "\u3000" <= ch <= "\u303f" or "\uff00" <= ch <= "\uffef"
        for ch in text
    )


def _code_block(code: str, styles):
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Table, TableStyle

    code = _demojify(code)
    mono_ok = not _has_cjk(code)
    parts = []
    for ln in code.split("\n"):
        if not ln.strip():
            parts.append("&nbsp;")
            continue
        txt = esc(ln).replace(" ", "&nbsp;")
        if mono_ok:
            # Keep the block monospace only on lines the mono face can draw;
            # one stray symbol would otherwise become a tofu box.
            face = _face_for(ln, MONO_FONT)
            if face != MONO_FONT:
                mono_ok = False
                txt = f'<font face="{BODY_FONT}">{txt}</font>'
        parts.append(txt)
    body = "<br/>".join(parts)
    style = styles["code"] if mono_ok else styles["code_cjk"]
    p = Paragraph(body, style)
    t = Table([[p]], hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F5F5")),
                ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


# --------------------------------------------------------------------------
# Team section
# --------------------------------------------------------------------------
TODO_RE = re.compile(r"\bTODO\b")


def load_team(path: Path) -> dict:
    import yaml

    raw = path.read_text(encoding="utf-8")

    # Keep the shipped template usable as documentation: strip comment lines
    # only for the purposes of *checking*, not for parsing.
    data = yaml.safe_load(raw) or {}
    return data


def find_todos(data: object, prefix: str = "") -> list[str]:
    out: list[str] = []
    if isinstance(data, dict):
        for k, v in data.items():
            out += find_todos(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(data, list):
        for idx, v in enumerate(data):
            out += find_todos(v, f"{prefix}[{idx}]")
    elif isinstance(data, str) and TODO_RE.search(data):
        out.append(prefix)
    elif data is None:
        out.append(f"{prefix} (null)")
    return out


def team_markdown(team: dict) -> str:
    name = team.get("team_name", "TODO")
    members = team.get("members") or []
    advisor = team.get("advisor") or {}
    lines = [f"# 团队介绍", ""]
    lines += [f"**队伍名称：** {name}", ""]
    lines += [f"**队伍规模：** {len(members)} 人（赛题上限 3 人）", ""]

    for idx, m in enumerate(members, 1):
        lines.append(f"## 成员 {idx}：{m.get('name','')}　—　{m.get('role','')}")
        lines.append("")
        lines.append(f"**单位：** {m.get('affiliation','')}　　**邮箱：** {m.get('email','')}")
        lines.append("")
        lines.append("### 相关研究 / 工程背景")
        lines.append("")
        lines.append(str(m.get("background", "")).rstrip())
        lines.append("")
        lines.append("### 过往竞赛 / 项目经验")
        lines.append("")
        lines.append(str(m.get("experience", "")).rstrip())
        lines.append("")
        lines.append("### 本次赛题分工")
        lines.append("")
        lines.append(str(m.get("duty", "")).rstrip())
        lines.append("")

    if advisor.get("name"):
        lines += [
            "## 指导老师",
            "",
            f"{advisor.get('name','')}　{advisor.get('title','')}　{advisor.get('affiliation','')}",
            "",
        ]

    lines += ["## 团队分工合理性与稳定性", "", str(team.get("team_statement", "")).rstrip(), ""]

    cr = team.get("compute_request") or {}
    lines += [
        "## 算力需求",
        "",
        "| 平台 | 申请卡日 |",
        "|---|---:|",
        f"| 天数智芯 BI-V150 | {cr.get('tianshu_bv150_card_days', 0)} |",
        f"| 沐曦曦云 C500 | {cr.get('metax_c500_card_days', 0)} |",
        "",
        str(cr.get("note", "")).rstrip(),
        "",
    ]
    commit = cr.get("commitment") or []
    if commit:
        lines += ["### 算力使用承诺", ""]
        for item in commit:
            lines.append(f"- {str(item).strip()}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------
def make_styles():
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm

    # allowWidows=0 stops a single carried-over line from being stranded alone
    # at the top of a page (reportlab permits it by default).
    # allowWidows=0 stops a single carried-over line from being stranded alone
    # at the top of a page (reportlab permits it by default).
    base = dict(fontName=BODY_FONT, wordWrap="CJK", leading=14.5, fontSize=9.5,
                allowWidows=0)
    return {
        "title": ParagraphStyle("title", fontName=BOLD_FONT, fontSize=17, leading=23,
                                spaceAfter=4, wordWrap="CJK"),
        # keepWithNext holds a heading back when its following block would not
        # fit, so a section title never sits alone at the foot of a page. It is
        # a look-ahead hint rather than a KeepTogether group: the follower may
        # still split across pages, so this never blows up the page count.
        "h1": ParagraphStyle("h1", fontName=BOLD_FONT, fontSize=14.5, leading=20,
                             spaceBefore=12, spaceAfter=4, wordWrap="CJK",
                             keepWithNext=1),
        "h2": ParagraphStyle("h2", fontName=BOLD_FONT, fontSize=12, leading=17,
                             spaceBefore=10, spaceAfter=3, wordWrap="CJK",
                             keepWithNext=1),
        "h3": ParagraphStyle("h3", fontName=BOLD_FONT, fontSize=10.5, leading=15,
                             spaceBefore=8, spaceAfter=2, wordWrap="CJK",
                             keepWithNext=1),
        "h4": ParagraphStyle("h4", fontName=BOLD_FONT, fontSize=9.8, leading=14,
                             spaceBefore=6, spaceAfter=2, wordWrap="CJK",
                             keepWithNext=1),
        "body": ParagraphStyle("body", alignment=TA_JUSTIFY, spaceAfter=2, **base),
        "bullet": ParagraphStyle("bullet", leftIndent=16, bulletIndent=4, spaceAfter=2,
                                 **base),
        "cell": ParagraphStyle("cell", fontName=BODY_FONT, fontSize=8.3, leading=11.6,
                               wordWrap="CJK"),
        "code": ParagraphStyle("code", fontName=MONO_FONT, fontSize=7.6, leading=10.6,
                               wordWrap="CJK"),
        "code_cjk": ParagraphStyle("code_cjk", fontName=BODY_FONT, fontSize=8.4,
                                   leading=12.4, wordWrap="CJK"),
        "quote": ParagraphStyle("quote", fontName=BODY_FONT, fontSize=9.2, leading=14,
                                leftIndent=10, textColor="#444444", wordWrap="CJK"),
    }


def render(md: str, out_path: Path, title: str, subtitle: str) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer

    styles = make_styles()
    doc = BaseDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=16 * mm,
        topMargin=17 * mm,
        bottomMargin=16 * mm,
        title=title,
        author="FlagOS S2 Track 2 submission",
        subject=subtitle,
    )

    def on_page(canv, d):
        canv.saveState()
        canv.setFont(BODY_FONT, 8)
        canv.setFillColorRGB(0.35, 0.35, 0.35)
        canv.drawCentredString(A4[0] / 2, 9 * mm, f"— {d.page} —")
        canv.drawString(18 * mm, A4[1] - 11 * mm, title)
        canv.drawRightString(A4[0] - 16 * mm, A4[1] - 11 * mm, subtitle)
        canv.setStrokeColorRGB(0.8, 0.8, 0.8)
        canv.line(18 * mm, A4[1] - 13 * mm, A4[0] - 16 * mm, A4[1] - 13 * mm)
        canv.restoreState()

    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=on_page)])

    story = build_story(md, styles)
    doc.build(story)


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="validate team.yaml and exit without rendering")
    ap.add_argument("--team", type=Path, default=TEAM_YAML,
                    help="team config to use (default: proposals/team.yaml)")
    ap.add_argument("--out", type=Path, help="output PDF path")
    args = ap.parse_args(argv)

    team_path: Path = args.team
    if not TECH_PLAN.exists():
        print(f"error: missing {TECH_PLAN}", file=sys.stderr)
        return 1
    if not team_path.exists():
        print(f"error: missing {team_path}", file=sys.stderr)
        return 1

    team = load_team(team_path)
    todos = find_todos(team)
    if todos:
        print("team.yaml still has unfilled fields:", file=sys.stderr)
        for t in todos:
            print(f"  - {t}", file=sys.stderr)
        print("\nFill proposals/team.yaml before generating the PDF.", file=sys.stderr)

    if args.check:
        return 1 if todos else 0
    if todos:
        return 1

    team_name = str(team["team_name"]).strip()
    out = args.out or (PROPOSALS / f"算力申请-{team_name}.pdf")

    md = TECH_PLAN.read_text(encoding="utf-8")
    md += "\n\n"
    md += team_markdown(team)

    register_fonts()
    install_cjk_kinsoku()
    render(md, out, title=f"算力申请 · {team_name}", subtitle="FlagOS S2 · 赛道二")

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"wrote {out}  ({size_mb:.2f} MB)")
    if size_mb > 20:
        print("WARNING: exceeds the 20 MB limit stated in the rules", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())