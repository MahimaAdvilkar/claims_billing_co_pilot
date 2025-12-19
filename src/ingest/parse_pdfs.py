import pdfplumber
from pathlib import Path
from typing import List, Dict

def _normalize_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = text.replace(" ,", ",")
    return " ".join(text.split())

def parse_pdf(file_path: str) -> List[Dict]:
    chunks = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            raw = page.extract_text() or ""
            text = _normalize_text(raw)
            if not text:
                continue
            chunks.append({
                "text": text,
                "metadata": {
                    "source": Path(file_path).name,
                    "page": i + 1,
                    "type": "pdf"
                }
            })
    return chunks

if __name__ == "__main__":
    from pprint import pprint
    sample = parse_pdf("data/raw/sample_invoice.pdf")
    pprint(sample[:2])