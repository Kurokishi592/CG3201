from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def outline_from(text: str) -> str:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    lines = [ln for ln in lines if "Copyright National University of Singapore" not in ln]
    if not lines:
        return "(no text)"
    return " / ".join(lines[:3])


def main() -> None:
    pdf_path = Path(
        r"c:\Users\Kenneth\Desktop\cg3201\Lab 2\Lecture 3 Bayesian Learning and Modelling.pdf"
    )
    out_text = pdf_path.with_name("Lecture 3 Bayesian Learning and Modelling.extracted.md")
    out_outline = pdf_path.with_name("Lecture 3 Bayesian Learning and Modelling.outline.md")

    reader = PdfReader(str(pdf_path))

    full_parts: list[str] = []
    outline_lines: list[str] = ["# Slide-by-slide outline", ""]

    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = clean_text(raw)

        full_parts.append(f"# Slide {i}\n\n{text}\n")
        outline_lines.append(f"- Slide {i}: {outline_from(text)}")

    out_text.write_text("\n".join(full_parts), encoding="utf-8")
    out_outline.write_text("\n".join(outline_lines) + "\n", encoding="utf-8")

    print(f"pages: {len(reader.pages)}")
    print(f"wrote: {out_text}")
    print(f"wrote: {out_outline}")


if __name__ == "__main__":
    main()
