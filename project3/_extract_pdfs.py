from pathlib import Path
from pypdf import PdfReader

root = Path(r"c:/Users/Kenneth/Desktop/cg3201")
out_dir = root / "project3" / "_extracted"
out_dir.mkdir(exist_ok=True)

pdfs = [
    root / "project3" / "Project 3_corrected.pdf",
    root / "project3" / "Lecture 5 Convolution Neural Network_updated_annotated.pdf",
    root / "project3" / "Lecture 6 CNN Architectures_updated_annotated.pdf",
    root / "Lab 3" / "CG3201 Lab3 Manual.pdf",
    root / "Lab 3" / "Lecture 5 Convolution Neural Network_updated.pdf",
]

for pdf in pdfs:
    reader = PdfReader(str(pdf))
    out = []
    out.append(f"# FILE: {pdf.name}")
    out.append(f"# PAGES: {len(reader.pages)}")
    for i, page in enumerate(reader.pages, start=1):
        out.append(f"\n\n===== PAGE {i} TEXT =====")
        text = page.extract_text() or ""
        out.append(text)

        ann_texts = []
        annots = page.get("/Annots")
        if annots:
            for ann_ref in annots:
                try:
                    ann = ann_ref.get_object()
                except Exception:
                    continue
                c = ann.get("/Contents")
                if c:
                    ann_texts.append(str(c))
                rc = ann.get("/RC")
                if rc:
                    ann_texts.append(str(rc))
                t = ann.get("/T")
                if t:
                    ann_texts.append(f"Author: {t}")
        if ann_texts:
            out.append(f"\n===== PAGE {i} ANNOTATIONS =====")
            out.extend(ann_texts)

    out_path = out_dir / (pdf.stem + ".txt")
    out_path.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {out_path}")
