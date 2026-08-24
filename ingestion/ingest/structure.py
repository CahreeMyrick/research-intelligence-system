"""
Stage 3: Document structure reconstruction.

Turns the flat list of TextBlocks from stage 2 into a tree of Sections,
each holding its own Paragraphs. This is heuristic (font-size + boldness +
numbering-pattern based) rather than ML-based -- it is deliberately simple,
and is the natural place to later swap in a stronger heading classifier
(e.g. GROBID) without touching the rest of the pipeline.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

from ingest.extraction import RawDocument, TextBlock
from ingest.schema import Paragraph, Section

# Headings that conventionally appear as their own top-level line in a
# paper, even when they aren't numbered ("Abstract", "References", ...).
KNOWN_TOP_LEVEL_HEADINGS = {
    "abstract",
    "introduction",
    "related work",
    "background",
    "methods",
    "methodology",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "acknowledgments",
    "acknowledgements",
    "references",
    "bibliography",
    "appendix",
    "limitations",
}

NUMBERED_HEADING_RE = re.compile(
    r"^(?P<num>\d+(?:\.\d+){0,3})\.?\s+(?P<title>[A-Z][^\.]{1,120})$"
)

MAX_HEADING_WORDS = 12


def _looks_like_heading(block: TextBlock, body_size: float) -> tuple[bool, int, str] | None:
    """
    Returns (is_heading, level, clean_title) if `block` looks like a
    section heading, else None.
    """
    text = block.text.strip()
    if not text or len(text.split()) > MAX_HEADING_WORDS:
        return None

    # Pattern 1: explicit numbering, e.g. "3.1 Discriminator"
    m = NUMBERED_HEADING_RE.match(text)
    if m:
        level = m.group("num").count(".") + 1
        title = f"{m.group('num')} {m.group('title')}".strip()
        return True, level, title

    # Pattern 2: known section name on its own line (case-insensitive),
    # optionally followed by a stray colon.
    normalized = text.rstrip(":").strip().lower()
    if normalized in KNOWN_TOP_LEVEL_HEADINGS:
        return True, 1, text.rstrip(":").strip()

    # Pattern 3: visually distinct (bold, and/or *meaningfully* larger than
    # body text -- a relative jump, not just a point or two, to avoid
    # flagging things like an 11pt byline against 10pt body text) and
    # short -- treat as a heading, level inferred from font size below.
    size_jump = block.size >= body_size * 1.15 and block.size - body_size >= 1.5
    if (block.bold or size_jump) and len(text.split()) <= 8:
        # Filter out obvious non-headings: body text fragments rarely lack
        # terminal punctuation AND start with a capital AND are short, but
        # to be safe, require it not end mid-sentence with a comma.
        if not text.endswith((",", ";")):
            return True, None, text  # level resolved later by font-size clustering

    return None


@dataclass
class _StackFrame:
    level: int
    section: Section


def _merge_wrapped_titles(blocks: list[TextBlock], body_size: float) -> list[TextBlock]:
    """
    A wrapped title/heading (e.g. "ViTGAN: Generative Adversarial Networks"
    / "with Vision Transformers") often comes back from PyMuPDF as two
    separate blocks rather than one, since PDFs have no native paragraph
    markup. Merge consecutive short, similarly-oversized blocks on the same
    page into a single logical heading block so they don't get split into
    two sections.
    """
    if not blocks:
        return blocks

    merged: list[TextBlock] = []
    i = 0
    while i < len(blocks):
        current = blocks[i]
        is_big = current.size - body_size >= 1.5
        if is_big and len(current.text.split()) <= 8:
            j = i + 1
            combined_text = current.text
            while (
                j < len(blocks)
                and blocks[j].page == current.page
                and abs(blocks[j].size - current.size) < 0.5
                and len(blocks[j].text.split()) <= 8
                and not blocks[j].text.strip().endswith((".", ":"))  # heuristic paragraph end
            ):
                combined_text += " " + blocks[j].text
                j += 1
            merged.append(
                TextBlock(
                    text=combined_text,
                    page=current.page,
                    size=current.size,
                    bold=current.bold,
                    order=current.order,
                )
            )
            i = j
        else:
            merged.append(current)
            i += 1
    return merged


def reconstruct(raw: RawDocument) -> list[Section]:
    """Build the top-level list of Sections (with nested subsections)."""
    if not raw.blocks:
        return []

    body_size = _estimate_body_size(raw.blocks)
    blocks = _merge_wrapped_titles(raw.blocks, body_size)

    # First pass: classify every block as heading-or-not, and for headings
    # with an unresolved level (Pattern 3), bucket by font size to resolve
    # level via clustering (largest distinct size = level 1, etc).
    classified: list[tuple[TextBlock, tuple[bool, int | None, str] | None]] = []
    unresolved_sizes: set[float] = set()

    for block in blocks:
        verdict = _looks_like_heading(block, body_size)
        classified.append((block, verdict))
        if verdict and verdict[1] is None:
            unresolved_sizes.add(block.size)

    size_rank = {
        size: rank + 1
        for rank, size in enumerate(sorted(unresolved_sizes, reverse=True))
    }

    root: list[Section] = []
    stack: list[_StackFrame] = []
    para_position = 0

    def current_container() -> list[Section]:
        return stack[-1].section.subsections if stack else root

    for block, verdict in classified:
        if verdict:
            is_heading, level, title = verdict
            if level is None:
                level = size_rank.get(block.size, 1)
            # Heuristic ceiling: don't let font-size clustering alone
            # produce implausibly deep nesting.
            level = min(level, 4)

            section = Section(title=title, level=level, page_start=block.page)

            while stack and stack[-1].level >= level:
                stack.pop()

            current_container().append(section)
            stack.append(_StackFrame(level=level, section=section))
        else:
            para = Paragraph(text=block.text, page=block.page, position=para_position)
            para_position += 1
            if stack:
                stack[-1].section.paragraphs.append(para)
            else:
                # Body text before any detected heading (e.g. a title-page
                # byline paragraph) -- park it in an implicit preamble
                # section so no content is silently dropped.
                preamble = _ensure_preamble(root)
                preamble.paragraphs.append(para)

    _fill_page_ends(root, raw.page_count)
    return root


def _estimate_body_size(blocks: list[TextBlock]) -> float:
    sizes = [b.size for b in blocks if len(b.text.split()) >= 6]
    sizes = sizes or [b.size for b in blocks]
    try:
        return statistics.mode([round(s) for s in sizes])
    except statistics.StatisticsError:
        return statistics.median(sizes)


def _ensure_preamble(root: list[Section]) -> Section:
    if root and root[0].title == "(preamble)":
        return root[0]
    preamble = Section(title="(preamble)", level=1)
    root.insert(0, preamble)
    return preamble


def _fill_page_ends(sections: list[Section], doc_page_count: int) -> None:
    """Best-effort page_end: the page the *next sibling or ancestor* starts on."""

    def walk(sec_list: list[Section], fallback_end: int):
        for i, sec in enumerate(sec_list):
            next_start = (
                sec_list[i + 1].page_start
                if i + 1 < len(sec_list) and sec_list[i + 1].page_start is not None
                else fallback_end
            )
            sec.page_end = next_start
            walk(sec.subsections, next_start)

    walk(sections, doc_page_count - 1 if doc_page_count else 0)
