"""
ocr_loader.py
PDF files se text extract karta hai.
- pdfplumber: tables ko sahi extract karta hai (course + fees kabhi mix nahi hogi)
- PyMuPDF: normal text ke liye backup
- pytesseract: scanned PDFs ke liye
"""

import os
import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path


def extract_tables_as_text(pdf_path: str) -> str:
    """
    pdfplumber se text + tables dono extract karo.
    Table rows intact rehti hain — course name aur fees kabhi mix nahi hogi.
    """
    all_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):

            # Normal text extract
            text = page.extract_text()
            if text and text.strip():
                all_text.append(f"\n--- Page {page_num + 1} ---\n{text.strip()}")

            # Tables extract — har row clearly labeled
            tables = page.extract_tables()
            for t_idx, table in enumerate(tables):
                if not table:
                    continue
                all_text.append(f"\n--- Table {t_idx + 1} on Page {page_num + 1} ---")
                for row in table:
                    if not row:
                        continue
                    clean = [str(cell).strip() if cell else "" for cell in row]
                    # Skip completely empty rows
                    if any(c for c in clean):
                        all_text.append(" | ".join(clean))

    return "\n".join(all_text)


def is_scanned_pdf(pdf_path: str, sample_pages: int = 3) -> bool:
    """
    Check karo ki PDF scanned hai ya normal text PDF.
    Agar pehle kuch pages mein text nahi mila → scanned maano.
    """
    doc = fitz.open(pdf_path)
    total = min(sample_pages, len(doc))
    text_found = 0

    for i in range(total):
        page = doc[i]
        text = page.get_text("text").strip()
        if len(text) > 50:
            text_found += 1

    doc.close()
    return text_found == 0


def extract_text_ocr(pdf_path: str) -> str:
    """
    Scanned PDF se text extract karo using pytesseract.
    Har page ko image mein convert karta hai, phir OCR.
    """
    try:
        import pytesseract
        from PIL import Image
        import io
    except ImportError:
        raise ImportError(
            "Scanned PDF ke liye install karo:\n"
            "pip install pytesseract Pillow\n"
            "aur Tesseract binary bhi: https://github.com/tesseract-ocr/tesseract"
        )

    doc = fitz.open(pdf_path)
    all_text = []

    for page_num, page in enumerate(doc):
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")

        from PIL import Image
        import io
        image = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(image, lang="eng")

        if text.strip():
            all_text.append(f"\n--- Page {page_num + 1} ---\n{text}")

        print(f"  OCR done: page {page_num + 1}/{len(doc)}")

    doc.close()
    return "\n".join(all_text)


def load_pdf(pdf_path: str) -> str:
    """
    Main function — PDF path do, text wapas milega.
    pdfplumber use karta hai tables ke liye.
    Scanned PDF ho toh pytesseract fallback.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF nahi mili: {pdf_path}")

    print(f"Loading: {pdf_path}")

    if is_scanned_pdf(pdf_path):
        print("  -> Scanned PDF detected, OCR use karega...")
        return extract_text_ocr(pdf_path)
    else:
        print("  -> Normal text PDF, pdfplumber se extract...")
        return extract_tables_as_text(pdf_path)


def load_all_pdfs(data_dir: str = "data") -> dict:
    """
    data/ folder ke saare PDFs load karo.
    Returns: {filename: extracted_text}
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"data/ folder nahi mila: {data_dir}")

    pdf_files = list(data_path.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"Koi PDF nahi mili {data_dir}/ mein")

    results = {}
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            text = load_pdf(str(pdf_file))
            results[pdf_file.name] = text
            print(f"  [ok] {len(text)} characters extracted")
        except Exception as e:
            print(f"  [error] Error in {pdf_file.name}: {e}")

    return results


if __name__ == "__main__":
    docs = load_all_pdfs("data")
    for name, text in docs.items():
        print(f"\n{name}: {len(text)} chars")
        print(text[:500])
