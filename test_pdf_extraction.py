import fitz
import os
import zipfile
import asyncio
from pdf_extraction import pdf_extraction

def create_test_pdf(filename, text):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((100, 100), text, fontsize=20)
    doc.save(filename)
    doc.close()
    print(f"📄 Created test PDF: {filename}")

def create_test_zip(zip_name, pdf_files):
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for pdf in pdf_files:
            zipf.write(pdf)
    print(f"📦 Created test ZIP: {zip_name}")

async def run_test():
    # 1. Create dummy PDFs with trigger words for AI classification
    create_test_pdf("test_transcript.pdf", "OFFICIAL TRANSCRIPT OF RECORDS")
    create_test_pdf("test_degree.pdf", "DEGREE CONFERRED UPON")
    
    # 2. Create a ZIP
    create_test_zip("test_bundle.zip", ["test_transcript.pdf", "test_degree.pdf"])
    
    # 3. Test pdf_extraction (now async and includes JSON extraction)
    print("\n--- Testing pdf_extraction with ZIP + AI EXTRACTION ---")
    await pdf_extraction("test_bundle.zip")

if __name__ == "__main__":
    asyncio.run(run_test())
