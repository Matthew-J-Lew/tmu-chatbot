from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from bs4 import BeautifulSoup, Tag

from app.api.program_registry import match_program


@dataclass
class SectionBlock:
    section: str
    text: str
    kind: str = "section"


@dataclass
class HtmlDocument:
    title: str
    blocks: List[SectionBlock]


NOISY_SELECTORS = [
    "nav",
    "footer",
    "aside",
    "script",
    "style",
    "noscript",
    "form",
    "header .global-header",
]
NOISY_KEYWORDS = {
    "cookie",
    "banner",
    "subscribe",
    "social",
    "share",
    "breadcrumb",
    "sidebar",
    "search",
    "skip-to",
    "directory maps",
    "follow us",
}
BUTTON_PATTERNS = [
    re.compile(r"^Explore program requirements for\b", re.I),
    re.compile(r"^Visit the Department of\b", re.I),
    re.compile(r"^How to apply$", re.I),
    re.compile(r"^Experience TMU$", re.I),
]
IMAGE_ALT_PATTERNS = [
    re.compile(r"^(?:A|An|The|Two|Three|Young|Students?)\b.+", re.I),
]
_CALENDAR_ARTS_RE = re.compile(
    r"/calendar/(?P<year>\d{4}-\d{4})/programs/arts/(?P<slug>[^/]+)(?:/(?P<table>table_i|table_ii))?/?$",
    re.I,
)


def _norm_ws(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n+ *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _text(el: Tag) -> str:
    return _norm_ws(el.get_text("\n", strip=True))


def _looks_like_image_alt(line: str) -> bool:
    line = line.strip()
    if len(line) < 20:
        return False
    return any(p.match(line) for p in IMAGE_ALT_PATTERNS)


def _looks_like_button_line(line: str) -> bool:
    line = line.strip()
    return any(p.match(line) for p in BUTTON_PATTERNS)


def _looks_like_noise_line(line: str) -> bool:
    lower = line.strip().lower()
    if not lower:
        return True
    if _looks_like_button_line(line):
        return True
    if _looks_like_image_alt(line):
        return True
    if lower in {"download the viewbook", "join the faculty of arts", "explore departments", "explore programs"}:
        return True
    return False


def _clean_lines(lines: Iterable[str]) -> List[str]:
    out: List[str] = []
    for line in lines:
        line = _norm_ws(line)
        if not line:
            continue
        if _looks_like_noise_line(line):
            continue
        out.append(line)
    return out


def _clean_text(text: str) -> str:
    return "\n".join(_clean_lines(text.splitlines()))


def _title_from_soup(soup: BeautifulSoup, base_url: str) -> str:
    title_text = None
    if soup.title and soup.title.string:
        title_text = _norm_ws(soup.title.string)
    return title_text or (urllib.parse.urlparse(base_url).path.strip("/") or "Untitled")


def _remove_noise(soup: BeautifulSoup) -> None:
    for sel in NOISY_SELECTORS:
        for tag in soup.select(sel):
            tag.decompose()

    for tag in list(soup.find_all(True)):
        classes = " ".join(tag.get("class") or [])
        ident = str(tag.get("id") or "")
        lowered = f"{classes} {ident}".lower()
        if any(k in lowered for k in NOISY_KEYWORDS):
            tag.decompose()


def _context_heading_for(node: Tag, default: str) -> Optional[str]:
    for sib in node.previous_siblings:
        if isinstance(sib, Tag):
            heading = sib.find(["h1", "h2", "h3", "h4"]) if sib.name not in {"h1", "h2", "h3", "h4"} else sib
            if heading:
                txt = _norm_ws(heading.get_text(" ", strip=True))
                if txt:
                    return txt
    parent = node.parent
    while isinstance(parent, Tag):
        prev = parent.previous_sibling
        while isinstance(prev, Tag):
            heading = prev.find(["h1", "h2", "h3", "h4"]) if prev.name not in {"h1", "h2", "h3", "h4"} else prev
            if heading:
                txt = _norm_ws(heading.get_text(" ", strip=True))
                if txt:
                    return txt
            prev = prev.previous_sibling
        parent = parent.parent
    return default


def _summary_text_for_container(container: Tag, context_heading: str) -> str:
    parts: List[str] = []
    prev_text_bits: List[str] = []
    for sib in container.previous_siblings:
        if not isinstance(sib, Tag):
            continue
        if sib.name in {"h1", "h2", "h3", "h4"}:
            break
        txt = _text(sib)
        if txt and txt != context_heading:
            prev_text_bits.append(txt)
    prev_text_bits.reverse()
    if prev_text_bits:
        parts.extend(prev_text_bits[-2:])

    titles = []
    for idx, panel in enumerate(container.select(":scope > .panel.panel-default"), 1):
        title_el = panel.select_one(".panel-heading .panel-title a") or panel.select_one(".panel-heading .panel-title")
        if not title_el:
            continue
        title = _norm_ws(title_el.get_text(" ", strip=True))
        if title:
            titles.append(f"{idx}. {title}")
    if titles:
        if context_heading and context_heading.lower() not in {"programs", "program list"}:
            parts.append(context_heading)
        parts.extend(titles)
    return _clean_text("\n".join(parts))


def _extract_rich_text_blocks(panel_body: Tag) -> List[str]:
    texts: List[str] = []
    for rich in panel_body.select(".c1 .resText .res-text, .c1 .resText.parbase.section .res-text"):
        txt = _clean_text(_text(rich))
        if txt:
            texts.append(txt)

    if not texts:
        for rich in panel_body.select(".resText .res-text, .resText.parbase.section .res-text"):
            parent_classes = " ".join(rich.parent.get("class") or []) if isinstance(rich.parent, Tag) else ""
            grandparent_classes = " ".join(rich.parent.parent.get("class") or []) if isinstance(rich.parent, Tag) and isinstance(rich.parent.parent, Tag) else ""
            if "c2" in parent_classes or "c2" in grandparent_classes:
                continue
            txt = _clean_text(_text(rich))
            if txt:
                texts.append(txt)

    for info in panel_body.select(".resInfographic .textContainer"):
        info_text = _clean_text(_text(info))
        if not info_text:
            continue
        info_words = info_text.split()
        if len(info_words) <= 8 and re.fullmatch(r"[\d$,+–\- ]+", info_text.splitlines()[0]):
            continue
        texts.append(f"At a glance: {info_text}")

    for rich in panel_body.select(".c2 .resText .res-text, .c2 .resText.parbase.section .res-text"):
        txt = _clean_text(_text(rich))
        if txt:
            texts.append(f"Student perspective: {txt}")
            break

    seen = set()
    deduped: List[str] = []
    for item in texts:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _extract_accordion_blocks(main: Tag, page_title: str) -> List[SectionBlock]:
    blocks: List[SectionBlock] = []
    for container in list(main.select(".panel-group.accordion")):
        context_heading = _context_heading_for(container, page_title) or page_title
        summary_text = _summary_text_for_container(container, context_heading)
        if summary_text:
            blocks.append(SectionBlock(section=context_heading, text=summary_text, kind="accordion_summary"))
        for panel in container.select(":scope > .panel.panel-default"):
            title_el = panel.select_one(".panel-heading .panel-title a") or panel.select_one(".panel-heading .panel-title")
            if not title_el:
                continue
            title = _norm_ws(title_el.get_text(" ", strip=True))
            if not title:
                continue
            panel_body = panel.select_one(".panel-collapse .panel-body") or panel.select_one(".panel-body")
            if not panel_body:
                continue
            text_parts = _extract_rich_text_blocks(panel_body)
            if not text_parts:
                continue
            text = _clean_text("\n\n".join(text_parts))
            if text:
                blocks.append(SectionBlock(section=title, text=text, kind="accordion_panel"))
        container.decompose()
    return blocks


def _extract_generic_sections(main: Tag, page_title: str) -> List[SectionBlock]:
    blocks: List[SectionBlock] = []
    current_heading: Optional[str] = None
    current_lines: List[str] = []

    def flush() -> None:
        nonlocal current_lines, current_heading
        text = _clean_text("\n".join(current_lines))
        if text:
            blocks.append(SectionBlock(section=current_heading or page_title, text=text, kind="generic_section"))
        current_lines = []

    for el in main.descendants:
        if not isinstance(el, Tag):
            continue
        if el.name in {"h1", "h2", "h3", "h4"}:
            heading = _norm_ws(el.get_text(" ", strip=True))
            if heading:
                flush()
                current_heading = heading
            continue
        if el.name in {"p", "li", "blockquote"}:
            txt = _clean_text(_text(el))
            if txt:
                current_lines.append(txt)
    flush()
    return blocks


def _clean_calendar_title(title: str) -> str:
    cleaned = _norm_ws(title)
    cleaned = re.sub(r"\s*-\s*Toronto Metropolitan University.*$", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*-\s*TMU.*$", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*-\s*Undergraduate Calendar.*$", "", cleaned, flags=re.I)
    return cleaned.strip(" -")


def _program_name_from_slug_or_title(slug: str, clean_title: str) -> str:
    for candidate in (clean_title, slug.replace("_", " ").replace("-", " ")):
        matched = match_program(candidate)
        if matched:
            return matched
    humanized = slug.replace("_", " ").replace("-", " ").title()
    return humanized


def _contextualize_calendar_block(base_url: str, title: str, section: str, text: str) -> Tuple[str, str]:
    match = _CALENDAR_ARTS_RE.search(urllib.parse.urlparse(base_url).path)
    if not match:
        return section, text

    year = match.group("year")
    slug = match.group("slug") or ""
    table = match.group("table")
    clean_title = _clean_calendar_title(title)
    program_name = _program_name_from_slug_or_title(slug, clean_title)
    page_context = clean_title or program_name
    section_clean = _norm_ws(section or "")

    if table:
        contextual_section = page_context if not section_clean or section_clean.lower() == page_context.lower() else f"{page_context} — {section_clean}"
    else:
        if not section_clean:
            contextual_section = page_context
        elif section_clean.lower().startswith(program_name.lower()):
            contextual_section = section_clean
        elif section_clean.lower() == page_context.lower():
            contextual_section = page_context
        else:
            contextual_section = f"{program_name} — {section_clean}"

    header_lines = [f"Academic year: {year}", f"Program: {program_name}"]
    if contextual_section and contextual_section.lower() not in text.lower():
        header_lines.append(f"Section context: {contextual_section}")
    contextual_text = _norm_ws("\n".join(header_lines) + "\n\n" + text)
    return contextual_section, contextual_text


def flatten_document_text(doc: HtmlDocument) -> str:
    return "\n\n".join(block.text for block in doc.blocks if block.text).strip()


def extract_html_document(html: str, base_url: str) -> HtmlDocument:
    soup = BeautifulSoup(html, "lxml")
    _remove_noise(soup)
    title = _title_from_soup(soup, base_url)
    main = soup.find("main") or soup.body or soup

    blocks: List[SectionBlock] = []
    blocks.extend(_extract_accordion_blocks(main, title))
    blocks.extend(_extract_generic_sections(main, title))

    cleaned_blocks: List[SectionBlock] = []
    seen_pairs = set()
    for block in blocks:
        section, block_text = _contextualize_calendar_block(base_url, title, block.section, block.text)
        sec = _norm_ws(section)[:240] or title
        txt = _clean_text(block_text)
        if not txt:
            continue
        if len(txt.split()) < 4:
            continue
        key = (sec.lower(), txt.lower())
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        cleaned_blocks.append(SectionBlock(section=sec, text=txt, kind=block.kind))
    return HtmlDocument(title=title, blocks=cleaned_blocks)
