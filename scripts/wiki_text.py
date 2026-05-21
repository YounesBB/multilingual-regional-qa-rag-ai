"""Helpers for extracting plain text from rendered Wikipedia HTML."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser


class RenderedTextExtractor(HTMLParser):
    SKIP_TAGS = {"script", "style", "noscript", "math"}
    BLOCK_TAGS = {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "caption",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        if data.strip():
            self.parts.append(data)


def rendered_html_to_text(rendered_html: str) -> str:
    parser = RenderedTextExtractor()
    parser.feed(rendered_html)
    parser.close()
    text = html.unescape("".join(parser.parts))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if lines and lines[-1]:
                lines.append("")
            continue
        if stripped in {"editovat", "editovat zdroj", "upravit"}:
            continue
        lines.append(stripped)
    return "\n".join(lines).strip()
