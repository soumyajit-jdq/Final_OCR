import os
import json
import re
import asyncio
import httpx
import base64
import io
import logging
import fitz
import anyio
import zipfile
import shutil
import tempfile
from PIL import Image
from web3 import Web3
from collections import OrderedDict
from dotenv import load_dotenv
from models import MarkSheetData, ValidationResponse, TranscriptData, CertificateData
from preprocessing import validate_image_quality

load_dotenv()

# Setup Logging
logger = logging.getLogger(__name__)

# CONFIG
OCR_API_KEY = os.getenv("OCR_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
# LEDGER_API_URL = os.getenv("LEDGER_API_URL", "http://localhost:5000/api/v1/ledger/upload") # Placeholder

class ProcessingService:
    @staticmethod
    async def validate_document(file_bytes: bytes, filename: str) -> ValidationResponse:
        """Runs the preprocessing quality checks in a separate thread to avoid blocking."""
        def sync_validate():
            is_valid, msg = validate_image_quality(file_bytes, filename)
            file_type = "PDF" if file_bytes.startswith(b"%PDF") else "Image"
            return is_valid, msg, file_type
            
        is_valid, msg, file_type = await anyio.to_thread.run_sync(sync_validate)
        return ValidationResponse(is_valid=is_valid, instruction=msg, file_type=file_type)

    @staticmethod
    async def compress_image(image_bytes: bytes, max_kb: int = 1000):
        """High-resolution compression for OCR.space (1MB limit)."""
        def sync_compress():
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            # If already small enough, don't touch it
            if len(image_bytes) <= max_kb * 1024:
                return image_bytes
                
            # Try to save with high quality first
            quality = 90
            buffer = io.BytesIO()
            while quality > 10:
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=quality, optimize=True)
                if len(buffer.getvalue()) <= max_kb * 1024:
                    logger.info(f"Image compressed to {len(buffer.getvalue())//1024}KB at quality {quality}")
                    return buffer.getvalue()
                quality -= 10
            img.thumbnail((1600, 1600))
                
            # If still too large, resize slightly (preserving resolution as much as possible)
            # img.thumbnail((2500, 2500))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=20)
            return buffer.getvalue()
            
        return await anyio.to_thread.run_sync(sync_compress)

    @staticmethod
    async def run_ocr(image_bytes: bytes):
        """Asynchronous call to OCR.space API."""
        compressed_bytes = await ProcessingService.compress_image(image_bytes)
        url = "https://api.ocr.space/parse/image"
        
        # Prepare multipart/form-data
        files = {"file": ("image.jpg", compressed_bytes, "image/jpeg")}
        data = {"apikey": OCR_API_KEY, "language": "eng", "isTable": True, "OCREngine": 2}
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, files=files, data=data, timeout=60)
                result = response.json()
                if result.get("OCRExitCode") != 1:
                    return f"OCR Failed: {result.get('ErrorMessage')}"
                return result["ParsedResults"][0]["ParsedText"]
            except Exception as e:
                return f"OCR Error: {e}"

    @staticmethod
    def encode_image(image_bytes):
        return base64.b64encode(image_bytes).decode('utf-8')

    @staticmethod
    def build_canonical_payload(data: dict) -> str:
        """
        Builds a canonical JSON string for Marksheets.
        """
        def clean(val):
            if val is None or str(val).lower() == "none":
                return ""
            return str(val).strip()

        subjects = []
        for s in data.get("subjects", []):
            ordered_subject = OrderedDict([
                ("code", clean(s.get("code"))),
                ("title", clean(s.get("title"))),
                ("credit_points", clean(s.get("credit_points")))
            ])
            subjects.append(ordered_subject)

        payload = OrderedDict([
            ("registration_no", clean(data.get("registration_no"))),
            ("name", clean(data.get("name"))),
            ("gpa", clean(data.get("gpa"))),
            ("subjects", subjects)
        ])
        return json.dumps(payload, separators=(',', ':'))

    @staticmethod
    def build_transcript_canonical_payload(data: dict) -> str:
        """
        Builds a canonical JSON string for the nested transcript structure.
        Omit missing or empty fields entirely to ensure no hardcoded data enters the hash.
        """
        def clean(val):
            if val is None or str(val).lower() == "none" or str(val).strip() == "":
                return None
            return str(val).strip()

        years = []
        for y in data.get("years", []) or []:
            semesters = []
            for s in y.get("semesters", []) or []:
                courses = []
                for c in s.get("courses", []) or []:
                    course_payload = OrderedDict()
                    c_num = clean(c.get("course_number"))
                    c_title = clean(c.get("title"))
                    c_points = clean(c.get("credit_points"))
                    
                    if c_num: course_payload["course_number"] = c_num
                    if c_title: course_payload["title"] = c_title
                    if c_points: course_payload["credit_points"] = c_points
                    
                    if course_payload:
                        courses.append(course_payload)
                
                sem_payload = OrderedDict()
                s_name = clean(s.get("semester"))
                s_gpa = clean(s.get("gpa"))
                s_cgpa = clean(s.get("cgpa"))
                
                if s_name: sem_payload["semester"] = s_name
                if s_gpa: sem_payload["gpa"] = s_gpa
                if s_cgpa: sem_payload["cgpa"] = s_cgpa
                if courses: sem_payload["courses"] = courses
                
                if sem_payload:
                    semesters.append(sem_payload)
            
            year_payload = OrderedDict()
            y_name = clean(y.get("year"))
            if y_name: year_payload["year"] = y_name
            if semesters: year_payload["semesters"] = semesters
            
            if year_payload:
                years.append(year_payload)
        
        top_courses = []
        for c in data.get("courses", []) or []:
            course_payload = OrderedDict()
            c_num = clean(c.get("course_number"))
            c_title = clean(c.get("title"))
            c_points = clean(c.get("credit_points"))
            if c_num: course_payload["course_number"] = c_num
            if c_title: course_payload["title"] = c_title
            if c_points: course_payload["credit_points"] = c_points
            if course_payload:
                top_courses.append(course_payload)

        payload = OrderedDict()
        reg_no = clean(data.get("registration_no"))
        name = clean(data.get("name"))
        degree = clean(data.get("degree"))
        adm_year = clean(data.get("admission_year"))
        comp_year = clean(data.get("completion_year"))
        ogpa = clean(data.get("ogpa"))
        result = clean(data.get("result"))
        class_div = clean(data.get("class_division"))

        if reg_no: payload["registration_no"] = reg_no
        if name: payload["name"] = name
        if degree: payload["degree"] = degree
        if adm_year: payload["admission_year"] = adm_year
        if comp_year: payload["completion_year"] = comp_year
        if ogpa: payload["ogpa"] = ogpa
        if result: payload["result"] = result
        if class_div: payload["class_division"] = class_div
        if years: payload["years"] = years
        if top_courses: payload["courses"] = top_courses

        return json.dumps(payload, separators=(',', ':'))

    @staticmethod
    def build_certificate_canonical_payload(data: dict) -> str:
        """
        Builds a canonical JSON string for academic certificates.
        Ensures strict parity with frontend hashing logic.
        """
        def clean(val):
            if val is None or str(val).lower() == "none":
                return ""
            return str(val).strip()

        payload = OrderedDict([
            ("certificate_no", clean(data.get("certificate_no"))),
            ("no", clean(data.get("no"))),
            ("name", clean(data.get("name"))),
            ("degree", clean(data.get("degree"))),
            ("branch", clean(data.get("branch"))),
            ("ogpa", clean(data.get("ogpa"))),
            ("year", clean(data.get("year"))),
            ("date", clean(data.get("date"))),
            ("class_division", clean(data.get("class_division")))
        ])
        return json.dumps(payload, separators=(',', ':'))

    @staticmethod
    async def process_pdf_pages(pdf_bytes: bytes, max_pages: int = 50):
        """High-resolution PDF processing (300 DPI)."""
        def sync_pdf_process():
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                num_pages = min(len(doc), max_pages)
                if num_pages == 0:
                    return [], ""
                
                all_text = []
                all_images = []
                for i in range(num_pages):
                    page = doc[i]
                    all_text.append(page.get_text().strip())
                    # Matrix(4, 4) ~ 288 DPI. tobytes("jpg") handles the rest.
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    all_images.append(pix.tobytes("jpg"))
                doc.close()
                return all_images, "\n\n".join(all_text)
            except Exception as e:
                logger.error(f"PDF processing failed: {e}")
                return [], ""
        
        return await anyio.to_thread.run_sync(sync_pdf_process)

    @staticmethod
    async def generate_keccak256(text: str):
        """Generates an Ethereum-standard Keccak-256 hash using Web3.py."""
        def sync_hash():
            hash_bytes = Web3.keccak(text=text)
            return Web3.to_hex(hash_bytes)
        return await anyio.to_thread.run_sync(sync_hash)

    @staticmethod
    async def generate_ledger_hash(structured_data: dict, doc_type: str) -> str:
        """
        Routing logic to build the canonical payload and hash it.
        This hash is used for manual anchoring.
        """
        try:
            if doc_type == "marksheet":
                payload = ProcessingService.build_canonical_payload(structured_data)
            elif doc_type == "transcript":
                payload = ProcessingService.build_transcript_canonical_payload(structured_data)
            elif doc_type == "certificate":
                payload = ProcessingService.build_certificate_canonical_payload(structured_data)
            else:
                return "0x" + "0" * 64

            ledger_hash = await ProcessingService.generate_keccak256(payload)
            logger.info(f"Generated Ledger Hash for manual anchoring ({doc_type}): {ledger_hash}")
            return ledger_hash
        except Exception as e:
            logger.error(f"Hash generation failed for {doc_type}: {e}")
            return "0x" + "0" * 64

    @staticmethod
    async def generate_with_cerebras(prompt: str):
        """High-speed text-only extraction using Cerebras."""
        try:
            url = "https://api.cerebras.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {CEREBRAS_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # --- PASS 1: Initial Extraction ---
            payload_1 = {
                "model": "llama3.1-8b",
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a VERBATIM marksheet parser. Extract exactly as visible in OCR. Return ONLY JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.0
            }
            async with httpx.AsyncClient() as client:
                resp_1_raw = await client.post(url, headers=headers, json=payload_1, timeout=30)
                resp_1 = resp_1_raw.json()
                
            if "choices" not in resp_1:
                return None
            initial_json = resp_1["choices"][0]["message"]["content"]
            
            # --- PASS 2: Self-Correction Loop ---
            correction_system_prompt = (
                "You are a character-level QA auditor. Correct any JSON formatting issues and ensure alignment with OCR.\n"
                "Return ONLY the corrected JSON object."
            )
            correction_user_prompt = f"RAW OCR TEXT:\n{prompt}\n\nINITIAL JSON TO CORRECT:\n{initial_json}"
            
            payload_2 = {
                "model": "llama3.1-8b",
                "messages": [
                    {"role": "system", "content": correction_system_prompt},
                    {"role": "user", "content": correction_user_prompt}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.0
            }
            async with httpx.AsyncClient() as client:
                resp_2_raw = await client.post(url, headers=headers, json=payload_2, timeout=30)
                resp_2 = resp_2_raw.json()
            
            if "choices" in resp_2:
                corrected_content = resp_2["choices"][0]["message"]["content"]
                logger.info("Self-Correction loop completed.")
                return json.loads(corrected_content)
            
            return json.loads(initial_json)
        except Exception as e:
            logger.warning(f"Cerebras extraction/correction failed: {e}")
            return None

    @staticmethod
    async def classify_document(ocr_text: str) -> str:
        """Uses keywords first, then Cerebras (Llama) to categorize the document type."""
        text_lower = ocr_text.lower()
        
        # 1. Faster, more reliable Keyword Map
        # Priorities: Evaluation Report -> Marksheet (Strongest)
        # Semester presence -> Transcript
        # Degree Certificate -> Certificate
        transcript_triggers = ["semester", "sem ", "transcript", "academic record", "consolidated marks", "consolidated statement"]
        marksheet_triggers = ["marksheet", "evaluation report", "statement of marks", "grade card", "memo of marks", "result", "grade sheet", "provisional marks"]
        certificate_triggers = ["degree certificate", "conferred upon", "passing certificate", "provisional certificate", "degree of"]

        if "evaluation report" in text_lower:
            return "marksheet"
        if any(t in text_lower for t in transcript_triggers):
            return "transcript"
        if any(t in text_lower for t in marksheet_triggers):
            return "marksheet"
        if any(t in text_lower for t in certificate_triggers):
            return "certificate"

        # 2. Fallback to AI Classification
        if not CEREBRAS_API_KEY:
            return "unknown"
        try:
            url = "https://api.cerebras.ai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"}
            prompt = (
                "Identify the document type strictly based on scope:\n"
                "1. 'marksheet': A record for a single semester/year (e.g., 'Grade Card').\n"
                "2. 'transcript': A multi-page consolidated record of all years/semesters.\n"
                "3. 'certificate': A single-page document conferring a degree (e.g., 'Degree Certificate').\n\n"
                "Respond ONLY with one word: 'marksheet', 'certificate', or 'transcript'.\n\n"
                f"TEXT:\n{ocr_text[:4000]}"
            )
            payload = {
                "model": "llama3.1-8b",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0
            }
            async with httpx.AsyncClient() as client:
                res = await client.post(url, headers=headers, json=payload, timeout=30)
                content = res.json()["choices"][0]["message"]["content"].lower().strip()
                logger.info(f"AI Classification Raw Result: {content}")
                
                if "marksheet" in content: return "marksheet"
                if "certificate" in content: return "certificate"
                if "transcript" in content: return "transcript"
                return "unknown"
        except Exception as e:
            logger.error(f"Document classification failed: {e}")
            return "unknown"

    @staticmethod
    async def gemini_generate_with_retry(prompt: str, schema, images: list = None, retries: int = 5):
        """Helper to call Gemini with exponential backoff on 503 errors. Supports Multi-modal (Vision)."""
        from google import genai
        from google.genai import types
        
        model_name = 'gemini-3.1-flash-lite-preview'
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Prepare contents (text + optional images)
        contents = [prompt]
        if images:
            for img_bytes in images:
                contents.append(types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'))

        for attempt in range(retries):
            try:
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0.1
                    )
                )
                return json.loads(response.text)
            except Exception as e:
                error_msg = str(e)
                if ("503" in error_msg or "UNAVAILABLE" in error_msg) and attempt < retries - 1:
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"Gemini 503/Unavailable, retrying in {wait_time}s... (Attempt {attempt+1}/{retries})")
                    await anyio.sleep(wait_time)
                else:
                    raise e

    @staticmethod
    async def extract_with_ai(image_data, ocr_text: str):
        """Processes OCR text using Gemini (Priority) or Cerebras to produce structured JSON."""
        prompt = f"""
STRICT INSTRUCTION: You are a stateless, automated JSON parsing application. 
1. DO NOT use external knowledge or your own training data to fill fields.
2. DO NOT save, store, or remember any information from this request.
3. Extract data VERBATIM from the OCR text provided below.
4. If a field is not found in the text, return an empty string.
5. Your role is STRICTLY a text-to-JSON converter.

#### EXTRACTION RULES ####
- **Non-Credit Courses**: If a course is indicated as non-credit or has no numeric credit points (e.g., PGS 503), set "credit_points" to '---'.
- **Avoid NaN**: NEVER use the string 'NaN' for any field. Use '---' for missing or non-numeric points.

OCR TEXT:
{ocr_text}

JSON FORMAT:
{{
  "registration_no": "...",
  "name": "...",
  "gpa": "...",
  "subjects": [
    {{
      "code": "...", 
      "title": "...", 
      "credits": "...",
      "grade": "...",
      "credit_points": "..."
    }}
  ]
}}
Return ONLY the JSON.
"""

        # 1. Try Gemini First (LLM + Vision)
        if GEMINI_API_KEY:
            try:
                # Always provide images to Gemini if available for best context
                result = await ProcessingService.gemini_generate_with_retry(prompt, MarkSheetData, images=image_data)
                logger.info("Gemini Extraction successful (Multi-modal).")
                return result
            except Exception as e:
                logger.warning(f"Gemini Extraction failed: {e}. Falling back to Cerebras.")

        # 2. Try Cerebras Fallback
        if CEREBRAS_API_KEY:
            cerebras_result = await ProcessingService.generate_with_cerebras(prompt)
            if cerebras_result:
                return cerebras_result

        raise ValueError("AI Extraction failed.")

    @staticmethod
    async def extract_transcript_with_ai(image_data, ocr_text: str):
        """Processes Transcript OCR text using Gemini (Priority)."""
        prompt = f"""
STRICT INSTRUCTION: You are a stateless, automated JSON parsing application. 
1. DO NOT use external knowledge; extract ONLY from the text provided.
2. DO NOT save or log the data.
3. Your role is strictly mapping OCR text to a NESTED HIERARCHY.
4. Format Year and Semester as ALL CAPS WORDS.

#### FIELD EXTRACTION RULES ####
1. **Omit Missing Data**: If a Year, Semester, GPA, CGPA, or Course is NOT found, OMIT the key entirely. DO NOT hardcode or "force" a Year/Semester structure if it's not explicitly in the text.
2. **Hierarchy vs Flat List**: 
   - If the transcript has headings like "FIRST SEMESTER", group courses into `years` and `semesters`.
   - If the transcript is a flat list of courses (common in consolidated transcripts), extract them into the top-level `courses` list.
3. **Header & Summary**: Extract `registration_no`, `name`, `degree`, `admission_year`, `completion_year`, `ogpa`, `result`, and `class_division`.
4. **Vertical Row-Shifting (CRITICAL)**: 
   - Some semesters use "One-Row Up Offset". 
   - **Detection**: Look at the line containing the Semester Title (e.g., "THIRD SEMESTER 7.8").
     - If the Semester Title line ends with a number (e.g., "7.8" or "7.1"), an **OFFSET** is active for that entire semester.
     - **Mapping Rule**: The credit points for Course N are found on the line of Course N-1.
   - If no number is in the header, use "Same-Line Alignment".
5. **Orphan Lines**: If a course has no numeric points on its own line or the previous line, look at the nearest orphan line.
6. **Non-Credit Courses**: For courses with grade 'S' or missing numeric points, set "credit_points" to '---'.
7. **GPA/CGPA**: Extract "G.P.A." and "C.G.P.A." ONLY if explicitly printed.

#### FORMATTING ####
- If grouping: Use "FIRST YEAR", "FIRST SEMESTER", etc. in UPPERCASE.
- Roman Numerals: Convert numerical suffixes in titles: "Crop Production-1" -> "Crop Production-I".
- Course Codes: Remove all spaces: "Agron. 1.1" -> "Agron.1.1".

#### CHARACTER ACCURACY & TITLES ####
1. ROMAN NUMERALS: OCR misreads "1", "l", "|" as "I" and "ll", "11", "IT" as "II".
   - **ALWAYS** convert numerical suffixes to Roman: "Practical Crop Production-1" -> "Practical Crop Production-I", "Field Crops-ll" -> "Field Crops-II", "RAWE-l" -> "RAWE-I".
   - For `Pl.Phy.3.1`, ensure the title is "Crop Physiology - I".
2. COURSE CODES: Remove all spaces: "Agron. 1.1" -> "Agron.1.1". Correct prefixes: "LPM", "Ag.Econ", "Ag.Ento", "Pl.Phy", "Ag.Extn".
3. VERBATIM TITLES: Keep "Systamatics" exactly as spelled.

OCR TEXT:
{ocr_text}

JSON STRUCTURE:
{{
  "registration_no": "...",
  "name": "...",
  "degree": "...",
  "admission_year": "...",
  "completion_year": "...",
  "ogpa": "...",
  "result": "...",
  "class_division": "...",
  "years": [
    {{
      "year": "...",
      "semesters": [
        {{
            "semester": "...",
            "gpa": "...",
            "cgpa": "...",
            "courses": [
              {{ 
                "course_number": "...", 
                "title": "...", 
                "credits": "...",
                "grade": "...",
                "credit_points": "..." 
              }}
            ]
        }}
      ]
    }}
  ]
}}
Return ONLY JSON.
"""
        try:
            # Always use Vision for transcripts to ensure nested structure accuracy across all pages
            return await ProcessingService.gemini_generate_with_retry(prompt, TranscriptData, images=image_data)
        except Exception as e:
            logger.error(f"Transcript Extraction failed: {e}")
            raise e

    @staticmethod
    async def extract_certificate_with_ai(image_data, ocr_text: str):
        """Processes Certificate OCR text using Gemini (Priority)."""
        prompt = f"""
STRICT INSTRUCTION: You are a stateless, automated JSON parsing application. 
1. DO NOT use external knowledge.
2. Extract data VERBATIM from the text below.
3. Your ONLY task is converting OCR text into the specified JSON schema.
4. If a field is not found, return an empty string. NEVER use 'NaN'.

#### EXTRACTION RULES ####
- **Missing Numeric Fields**: If OGPA or other numeric fields are missing, use '---'.
- **OGPA Formatting**: Extract ONLY the numeric value (e.g., '8.12'). DO NOT include the scale or any suffix like ' / 10.00'.
- **Avoid NaN**: NEVER use the string 'NaN' for any field.

OCR TEXT:
{ocr_text}

JSON STRUCTURE:
{{
  "certificate_no": "...",
  "no": "...",
  "name": "...",
  "degree": "...",
  "branch": "...",
  "ogpa": "...",
  "year": "...",
  "date": "...",
  "class_division": "..."
}}
Return ONLY JSON.
"""
        try:
            # Ensure images are passed as a list
            images = image_data if isinstance(image_data, list) else [image_data]
            
            result = await ProcessingService.gemini_generate_with_retry(prompt, CertificateData, images=images)
            if result.get("ogpa"):
                # Clean up OGPA to remove any scale like "/ 10.00"
                result["ogpa"] = str(result["ogpa"]).split('/')[0].strip()
            return result
        except Exception as e:
            logger.error(f"Certificate Extraction failed: {e}")
            raise e


    @staticmethod
    async def process_zip(zip_bytes: bytes):
        """
        Processes a ZIP file containing multiple PDFs.
        Supports mixed-document PDFs (e.g. marksheets followed by certificates).
        Splits single PDFs into multiple logical sub-documents based on content.
        """
        results = []
        temp_dir = tempfile.mkdtemp()
        try:
            zip_path = os.path.join(temp_dir, "upload.zip")
            with open(zip_path, "wb") as f:
                f.write(zip_bytes)

            extract_path = os.path.join(temp_dir, "extracted")
            os.makedirs(extract_path, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

            pdf_files = []
            for root, _, files in os.walk(extract_path):
                for file in files:
                    if file.lower().endswith(".pdf"):
                        pdf_files.append(os.path.join(root, file))

            total_files = len(pdf_files)
            processed_count = 0
            failed_count = 0

            for pdf_path in pdf_files:
                filename = os.path.basename(pdf_path)
                logger.info(f"--- START PROCESSING: {filename} ---")
                try:
                    with open(pdf_path, "rb") as f:
                        file_bytes = f.read()

                    # 1. Validate
                    validation = await ProcessingService.validate_document(file_bytes, filename)
                    if not validation.is_valid:
                        results.append({
                            "filename": filename,
                            "doc_type": "unknown",
                            "status": "failed",
                            "error": f"Validation failed: {validation.instruction}"
                        })
                        failed_count += 1
                        continue

                    # 2. Extract images and text for ALL pages
                    img_list, _ = await ProcessingService.process_pdf_pages(file_bytes, max_pages=50)
                    
                    # 3. Parallel OCR & Classification
                    # Limit concurrency to 10 to avoid rate limits/overwhelming system
                    semaphore = asyncio.Semaphore(12)
                    
                    async def process_page(idx, img):
                        async with semaphore:
                            text = await ProcessingService.run_ocr(img)
                            doc_type = await ProcessingService.classify_document(text)
                            return idx, text, doc_type, img

                    tasks = [process_page(i, img) for i, img in enumerate(img_list)]
                    page_results = await asyncio.gather(*tasks)
                    # Sort results by index to maintain order
                    page_results.sort(key=lambda x: x[0])

                    # 4. Dynamic Grouping Logic
                    doc_groups = []
                    current_group = None
                    force_transcript = False

                    for i, text, page_type, img in page_results:
                        # 1. Text-based keyword overrides (Highest Priority)
                        clean_text = text.lower()
                        is_explicit_marksheet = "evaluation report" in clean_text or "marksheet" in clean_text
                        
                        if is_explicit_marksheet:
                            page_type = "marksheet"
                            force_transcript = False # Kill any pending force immediately
                        
                        # 2. Apply Heuristic for transcript
                        if force_transcript and not is_explicit_marksheet:
                            # Only force if it's not a certificate
                            if page_type != "certificate":
                                page_type = "transcript"
                                logger.info(f"Page {i+1}: Forced to TRANSCRIPT due to previous page heuristic.")
                            force_transcript = False
                        
                        # 3. Set flag for next iteration (Only if this page is a transcript)
                        if page_type == "transcript" or "transcript of academic record" in clean_text:
                            force_transcript = True

                        # 4. Final Rule: "If you don't find the word transcript then please don't give result in transcript."
                        if page_type == "transcript" and "transcript" not in clean_text:
                            page_type = "marksheet"
                            force_transcript = False

                        # SKIP UNKNOWN PAGES (except if forced)
                        if page_type == "unknown":
                            logger.info(f"Skipping page {i+1} of {filename}: Content not recognized.")
                            continue

                        print(f"\n[TERMINAL] OCR TEXT - PAGE {i+1} of {filename}:\n{text[:500]}...")

                        is_new_group = False
                        if not current_group:
                            is_new_group = True
                        elif current_group['type'] != page_type:
                            is_new_group = True
                        elif page_type in ['certificate', 'marksheet']:
                            if "evaluation report" in text.lower() or "certificate" in text.lower():
                                is_new_group = True

                        if is_new_group:
                            current_group = {
                                'type': page_type,
                                'pages': [i],
                                'images': [img],
                                'text': text
                            }
                            doc_groups.append(current_group)
                        else:
                            current_group['pages'].append(i)
                            current_group['images'].append(img)
                            current_group['text'] += "\n\n" + text

                    # 4. Process each detected sub-document group
                    if not doc_groups:
                        results.append({
                            "filename": filename,
                            "doc_type": "unknown",
                            "status": "failed",
                            "error": "No academic record identified (Marksheet/Certificate/Transcript) in this PDF."
                        })
                        failed_count += 1
                        continue

                    for idx, group in enumerate(doc_groups):
                        doc_type = group['type']
                        ocr_text = group['text']
                        images = group['images']
                        group_name = f"{filename} (Part {idx+1})" if len(doc_groups) > 1 else filename

                        logger.info(f"Extracting {doc_type} from {group_name}...")

                        structured_data = None
                        try:
                            if doc_type == "marksheet":
                                structured_data = await ProcessingService.extract_with_ai(images, ocr_text)
                            elif doc_type == "transcript":
                                structured_data = await ProcessingService.extract_transcript_with_ai(images, ocr_text)
                            elif doc_type == "certificate":
                                structured_data = await ProcessingService.extract_certificate_with_ai(images, ocr_text)
                            else:
                                raise ValueError("Unknown document type")

                            # 5. Generate Hash for Manual Anchoring
                            ledger_hash = await ProcessingService.generate_ledger_hash(structured_data, doc_type)
                            
                            results.append({
                                "filename": group_name,
                                "doc_type": doc_type,
                                "status": "success",
                                "data": structured_data,
                                "raw_text": ocr_text,
                                "ledger_hash": ledger_hash
                            })
                            processed_count += 1

                        except Exception as e:
                            logger.error(f"Error in group {idx+1} of {filename}: {e}")
                            results.append({
                                "filename": group_name,
                                "doc_type": doc_type,
                                "status": "failed",
                                "error": str(e)
                            })
                            failed_count += 1

                except Exception as e:
                    logger.error(f"Critical error processing {filename}: {e}")
                    results.append({
                        "filename": filename,
                        "doc_type": "unknown",
                        "status": "error",
                        "error": str(e)
                    })
                    failed_count += 1

            return {
                "total_files": total_files,
                "processed_files": processed_count,
                "failed_files": failed_count,
                "results": results
            }

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
