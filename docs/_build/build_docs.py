"""Build the specdiff docs into one self-contained HTML page.

Converts the five markdown files into panes of a single document, rewrites the
inter-document links to in-page anchors, and turns ```mermaid fences into the
<pre class="mermaid"> blocks the artifact runtime renders natively.
"""

from __future__ import annotations

import html
import json
import pathlib
import re

import markdown

ROOT = pathlib.Path("/Users/marcellobullo/research/specdiff")
OUT = pathlib.Path(__file__).parent / "specdiff-docs.html"

# slug -> (source path, nav title, who it is for)
DOCS = [
    ("overview", ROOT / "README.md", "Overview",
     "what this is, notation, quick start"),
    ("writing-a-verifier", ROOT / "docs/writing-a-verifier.md", "Writing a verifier",
     "implementing a coupling"),
    ("models", ROOT / "docs/models.md", "Plugging in your model",
     "target, schedule, proposals, backends"),
    ("architecture", ROOT / "docs/architecture.md", "Architecture",
     "how a round works, and why"),
    ("api-reference", ROOT / "docs/api-reference.md", "API reference",
     "every exported symbol"),
]

# source filename -> slug, for rewriting cross-document links
FILE_TO_SLUG = {
    "README.md": "overview",
    "../README.md": "overview",
    "docs/README.md": "overview",
    "writing-a-verifier.md": "writing-a-verifier",
    "docs/writing-a-verifier.md": "writing-a-verifier",
    "models.md": "models",
    "docs/models.md": "models",
    "architecture.md": "architecture",
    "docs/architecture.md": "architecture",
    "api-reference.md": "api-reference",
    "docs/api-reference.md": "api-reference",
}


def slugify(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text).lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s]+", "-", text.strip())


def convert(slug: str, path: pathlib.Path):
    raw = path.read_text()

    # protect mermaid fences from the markdown converter
    blocks: list[str] = []

    def stash(m):
        blocks.append(m.group(1))
        return f"\n\nMERMAIDBLOCK{len(blocks) - 1}ENDBLOCK\n\n"

    raw = re.sub(r"```mermaid\n(.*?)```", stash, raw, flags=re.S)

    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists", "attr_list"])
    out = md.convert(raw)

    # Rewrite links FIRST. The heading pass below inserts anchors that already
    # carry the doc prefix, so running it first would let the link rewriter
    # prefix them a second time (#api-reference--api-reference--backends).
    def link(m):
        target = m.group(1)
        if target.startswith(("http", "mailto")):
            return m.group(0) + ' target="_blank" rel="noopener"'
        if target.startswith("#"):
            return f'href="#{slug}--{target[1:]}"'
        file_part, _, frag = target.partition("#")
        dest = FILE_TO_SLUG.get(file_part)
        if dest is None:
            # a source file (examples/…, specdiff/…) — not a page we render
            return f'data-path="{target}" class="srclink"'
        return f'href="#{dest}--{frag}"' if frag else f'href="#{dest}"'

    out = re.sub(r'href="([^"]+)"', link, out)

    # unique, doc-prefixed heading ids + anchor links, and collect the TOC
    toc: list[dict] = []

    def head(m):
        level, inner = int(m.group(1)), m.group(2)
        anchor = slugify(inner)
        hid = f"{slug}--{anchor}"
        if level in (2, 3):
            toc.append({"id": hid, "text": re.sub(r"<[^>]+>", "", inner), "level": level})
        return (
            f'<h{level} id="{hid}" class="hd hd-{level}">'
            f'<a class="anchor" href="#{hid}" aria-label="Link to this section">#</a>'
            f"{inner}</h{level}>"
        )

    out = re.sub(r"<h([1-6])>(.*?)</h\1>", head, out, flags=re.S)

    # restore mermaid
    def pop(m):
        return f'<div class="mermaid-wrap"><pre class="mermaid">{html.escape(blocks[int(m.group(1))])}</pre></div>'

    out = re.sub(r"<p>MERMAIDBLOCK(\d+)ENDBLOCK</p>", pop, out)

    # A pipe inside a table cell has to be written \| in the source or it would
    # end the cell. GitHub unescapes it on render; python-markdown leaves the
    # backslash in, which turns ||mu_q - mu_p|| into \|\|mu_q - mu_p\|\|.
    out = re.sub(
        r"<(t[dh])>(.*?)</\1>",
        lambda m: f"<{m.group(1)}>{m.group(2).replace(chr(92) + '|', '|')}</{m.group(1)}>",
        out,
        flags=re.S,
    )

    # scroll containers for wide content
    out = out.replace("<table>", '<div class="tw"><table>').replace("</table>", "</table></div>")
    return out, toc


panes, navs = [], []
for slug, path, title, blurb in DOCS:
    body, toc = convert(slug, path)
    panes.append(f'<section class="pane" id="pane-{slug}" data-slug="{slug}">{body}</section>')
    navs.append({"slug": slug, "title": title, "blurb": blurb, "toc": toc})

nav_html = []
for n in navs:
    items = "".join(
        f'<a class="toc-item lvl{t["level"]}" href="#{t["id"]}">{html.escape(t["text"])}</a>'
        for t in n["toc"]
    )
    nav_html.append(
        f'<div class="nav-group" data-slug="{n["slug"]}">'
        f'<a class="nav-doc" href="#{n["slug"]}">'
        f'<span class="nav-title">{html.escape(n["title"])}</span>'
        f'<span class="nav-blurb">{html.escape(n["blurb"])}</span></a>'
        f'<div class="toc">{items}</div></div>'
    )

search_index = json.dumps(
    [{"s": n["slug"], "t": n["title"], "h": [{"i": t["id"], "x": t["text"]} for t in n["toc"]]}
     for n in navs]
)

page = (
    (pathlib.Path(__file__).parent / "shell.html").read_text()
    .replace("<!--NAV-->", "".join(nav_html))
    .replace("<!--PANES-->", "".join(panes))
    .replace("/*INDEX*/", search_index)
)

# Emit pure ASCII with everything else as numeric character references. The page
# is injected into a host <head>/<body> we do not control, so we cannot count on
# a charset declaration reaching the parser before this content does; entities
# render identically whatever encoding the document is read as.
OUT.write_text(page.encode("ascii", "xmlcharrefreplace").decode("ascii"))
print(f"wrote {OUT}  ({len(OUT.read_text()):,} bytes)")
for n in navs:
    print(f"  {n['slug']:20s} {len(n['toc']):3d} sections")
