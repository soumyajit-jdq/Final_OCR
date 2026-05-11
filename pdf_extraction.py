import os
import fitz  # PyMuPDF
import requests
import json
import asyncio
from PIL import Image
import io
import zipfile
from dotenv import load_dotenv
from service import ProcessingService

load_dotenv()

# 🔑 OCR.space API key
API_KEY = os.getenv("OCR_API_KEY", "K85146131088957")


# -----------------------------
# OCR using OCR.space
# -----------------------------
def ocr_space_image(image):
    url = "https://api.ocr.space/parse/image" #OCR

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG', quality=50)  # compress
    img_byte_arr = img_byte_arr.getvalue()

    response = requests.post(
        url,
        files={"file": ("image.jpg", img_byte_arr)},
        data={
            "apikey": API_KEY,
            "language": "eng"
        }
    )

    result = response.json()

    if result.get("IsErroredOnProcessing"):
        return ""

    if "ParsedResults" not in result:
        return ""

    return result["ParsedResults"][0]["ParsedText"]


# Classification is now handled by ProcessingService.classify_document in service.py


# -----------------------------
# Convert PDF → images
# -----------------------------
def pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []

    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap()

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append((i + 1, img))

    return pages


# -----------------------------
# MAIN
# -----------------------------

def unzip_folder(zip_file, extract_to):
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ Unzipped successfully into {extract_to}")
    except zipfile.BadZipFile:
        print(f"❌ {zip_file} is corrupted or not a zip file")
    except FileNotFoundError:
        print(f"❌ {zip_file} not found")
    except Exception as e:
        print(f"❌ Error unzipping: {e}")

async def pdf_extraction(input_path):
    output_folder = "output_docs"
    if input_path.lower().endswith(".zip"):
        print(f"📦 Detected ZIP file: {input_path}")
        extract_to = "temp_extracted"
        unzip_folder(input_path, extract_to)
        
        # Collect all PDF paths from the extracted folder
        pdf_list = []
        for root, _, files in os.walk(extract_to):
            # Skip macOS metadata directories
            if "__MACOSX" in root:
                continue
            for file in files:
                # Skip hidden/metadata files and check for .pdf extension
                if not file.startswith("._") and file.lower().endswith(".pdf"):
                    pdf_list.append(os.path.join(root, file))
        
        if not pdf_list:
            print("⚠️ No PDFs found in the zip folder.")
            return

        print(f"🔍 Found {len(pdf_list)} PDFs. Starting batch processing...\n")

        # Process each PDF one by one
        all_results = []
        for i, pdf_path in enumerate(pdf_list, 1):
            filename = os.path.basename(pdf_path)
            print(f"🚀 [{i}/{len(pdf_list)}] Processing: {filename}")
            result = await process_pdf_with_extraction(pdf_path, output_folder)
            all_results.append({
                "filename": filename,
                "data": result
            })
        
        # Save all results to a JSON file
        with open("extraction_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n💾 All results saved to extraction_results.json")
            
    elif input_path.lower().endswith(".pdf"):
        result = await process_pdf_with_extraction(input_path, output_folder)
        all_results = [{"filename": os.path.basename(input_path), "data": result}]
        with open("extraction_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
    else:
        print("❌ Unsupported file format. Please provide a .pdf or .zip file.")

async def process_pdf_with_extraction(input_pdf, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    pages = pdf_to_images(input_pdf)

    print(f"📄 Analyzing pages in {os.path.basename(input_pdf)}...")

    grouped_docs = []
    current_group = []
    current_type = None

    # STEP 1: Classify & Group
    for page_num, img in pages:
        text = ocr_space_image(img)
        # Use the advanced AI classification from service.py
        doc_type_raw = await ProcessingService.classify_document(text)
        doc_type = doc_type_raw.upper()

        if doc_type == current_type:
            current_group.append((page_num, img, text))
        else:
            if current_group:
                grouped_docs.append((current_type, current_group))
            current_group = [(page_num, img, text)]
            current_type = doc_type

    if current_group:
        grouped_docs.append((current_type, current_group))

    # STEP 2: Extraction and Display
    pdf_results = []
    print(f"\n--- Results for {os.path.basename(input_pdf)} ---")
    
    for doc_type, group in grouped_docs:
        if doc_type == "UNKNOWN":
            continue
            
        print(f"\n📑 Found Document Type: {doc_type}")
        
        # Combine text and prepare images for AI
        combined_text = "\n\n".join([item[2] for item in group])
        images_bytes = []
        for _, img, _ in group:
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            images_bytes.append(buf.getvalue())

        try:
            structured_json = {}
            if doc_type == "TRANSCRIPT":
                # Process large transcripts page-by-page to avoid AI truncation
                print(f"🔄 Processing {len(images_bytes)} transcript pages individually...")
                merged_transcript = None
                
                for idx, (img_bytes, page_text) in enumerate(zip(images_bytes, [p[2] for p in group])):
                    page_result = await ProcessingService.extract_transcript_with_ai([img_bytes], page_text)
                    
                    if merged_transcript is None:
                        merged_transcript = page_result
                    else:
                        # Merge years and semesters from this page into the main result
                        for year_data in page_result.get("years", []):
                            # Find if year already exists
                            existing_year = next((y for y in merged_transcript["years"] if y["year"] == year_data["year"]), None)
                            if existing_year:
                                for sem_data in year_data.get("semesters", []):
                                    # Find if semester already exists
                                    existing_sem = next((s for s in existing_year["semesters"] if s["semester"] == sem_data["semester"]), None)
                                    if existing_sem:
                                        existing_sem["courses"].extend(sem_data["courses"])
                                    else:
                                        existing_year["semesters"].append(sem_data)
                            else:
                                merged_transcript["years"].append(year_data)
                
                structured_json = merged_transcript
            elif doc_type == "MARKSHEET":
                structured_json = await ProcessingService.extract_with_ai(images_bytes, combined_text)
            elif doc_type == "CERTIFICATE":
                structured_json = await ProcessingService.extract_certificate_with_ai(images_bytes, combined_text)

            print(f"✅ {doc_type} JSON Output:")
            print(json.dumps(structured_json, indent=2))
            
            pdf_results.append({
                "type": doc_type,
                "content": structured_json
            })
        except Exception as e:
            print(f"❌ Failed to extract {doc_type} data: {e}")

    # --- STEP 3: CONSOLIDATE RESULTS ---
    consolidated = {}
    
    for res in pdf_results:
        dtype = res["type"]
        content = res["content"]
        
        if dtype not in consolidated:
            consolidated[dtype] = content
            continue
            
        # Merge logic for TRANSCRIPT
        if dtype == "TRANSCRIPT":
            main = consolidated[dtype]
            # Merge header info if missing
            for field in ["registration_no", "name", "degree", "ogpa", "class_division"]:
                if not main.get(field) and content.get(field):
                    main[field] = content[field]
            
            # Merge years
            for new_year in content.get("years", []):
                existing_year = next((y for y in main["years"] if y["year"] == new_year["year"]), None)
                if existing_year:
                    # Merge semesters
                    for new_sem in new_year.get("semesters", []):
                        existing_sem = next((s for s in existing_year["semesters"] if s["semester"] == new_sem["semester"]), None)
                        if existing_sem:
                            # Keep the semester with more courses, or merge them
                            if len(new_sem["courses"]) > len(existing_sem["courses"]):
                                existing_sem["courses"] = new_sem["courses"]
                                if new_sem.get("gpa"): existing_sem["gpa"] = new_sem["gpa"]
                        else:
                            existing_year["semesters"].append(new_sem)
                else:
                    main["years"].append(new_year)
        
        # Merge logic for CERTIFICATE
        elif dtype == "CERTIFICATE":
            main = consolidated[dtype]
            for field, val in content.items():
                if not main.get(field) and val:
                    main[field] = val

    # Convert back to list format
    final_results = [{"type": t, "content": c} for t, c in consolidated.items()]
    print(f"\n--- End of {os.path.basename(input_pdf)} (Consolidated {len(final_results)} records) ---\n")
    return final_results


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    input_file = "/Users/ritambiswas/Downloads/Archive 2.zip" 
    if os.path.exists(input_file):
        asyncio.run(pdf_extraction(input_file))
    else:
        print(f"❌ File not found: {input_file}")