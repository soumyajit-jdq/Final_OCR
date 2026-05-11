import os
import json
import re
import httpx
import base64
import io
import logging
import fitz
import anyio
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
        subjects = []
        for s in data.get("subjects", []):
            ordered_subject = OrderedDict([
                ("code", str(s.get("code", ""))),
                ("title", str(s.get("title", ""))),
                ("credit_points", str(s.get("credit_points", "")))
            ])
            subjects.append(ordered_subject)

        payload = OrderedDict([
            ("registration_no", str(data.get("registration_no", ""))),
            ("name", str(data.get("name", ""))),
            ("gpa", str(data.get("gpa", ""))),
            ("subjects", subjects)
        ])
        return json.dumps(payload, separators=(',', ':'))

    @staticmethod
    def build_transcript_canonical_payload(data: dict) -> str:
        """
        Builds a canonical JSON string for the nested transcript structure.
        Ensures stable hashing through strict key ordering.
        """
        years = []
        for y in data.get("years", []):
            semesters = []
            for s in y.get("semesters", []):
                courses = []
                for c in s.get("courses", []):
                    courses.append(OrderedDict([
                        ("course_number", str(c.get("course_number", ""))),
                        ("title", str(c.get("title", ""))),
                        ("credit_points", str(c.get("credit_points", "")))
                    ]))
                semesters.append(OrderedDict([
                    ("semester", str(s.get("semester", ""))),
                    ("gpa", str(s.get("gpa", ""))),
                    ("cgpa", str(s.get("cgpa", ""))),
                    ("courses", courses)
                ]))
            years.append(OrderedDict([
                ("year", str(y.get("year", ""))),
                ("semesters", semesters)
            ]))

        payload = OrderedDict([
            ("registration_no", str(data.get("registration_no", ""))),
            ("name", str(data.get("name", ""))),
            ("degree", str(data.get("degree", ""))),
            ("admission_year", str(data.get("admission_year", ""))),
            ("completion_year", str(data.get("completion_year", ""))),
            ("ogpa", str(data.get("ogpa", ""))),
            ("result", str(data.get("result", ""))),
            ("class_division", str(data.get("class_division", ""))),
            ("years", years)
        ])
        return json.dumps(payload, separators=(',', ':'))

    @staticmethod
    def build_certificate_canonical_payload(data: dict) -> str:
        """
        Builds a canonical JSON string for academic certificates.
        """
        payload = OrderedDict([
            ("certificate_no", str(data.get("certificate_no", ""))),
            ("no", str(data.get("no", ""))),
            # ("university", str(data.get("university", ""))),
            ("name", str(data.get("name", ""))),
            ("degree", str(data.get("degree", ""))),
            ("branch", str(data.get("branch", ""))),
            ("ogpa", str(data.get("ogpa", ""))),
            ("year", str(data.get("year", ""))),
            ("date", str(data.get("date", ""))),
            ("class_division", str(data.get("class_division", "")))
        ])
        return json.dumps(payload, separators=(',', ':'))

    @staticmethod
    async def process_pdf_pages(pdf_bytes: bytes, max_pages: int = 3):
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
        # Check certificates first because they often mention "Transcript" or "Marks" in titles
        certificate_triggers = ["conferred upon", "degree certificate", "passing certificate", "provisional certificate"]
        transcript_keywords = ["TRANSCRIPT", "PERMANENT RECORD", "CUMULATIVE", "OGPA", "ACADEMIC RECORD"]
        marksheet_keywords = ["MARKSHEET", "STATEMENT OF MARKS", "GRADE CARD", "REPORT CARD"]
        certificate_keywords = ["PROVISIONAL CERTIFICATE", "DEGREE CERTIFICATE", "CONVOCATION", "DIPLOMA", "CERTIFIES THAT"]

        if any(t in text_lower for t in certificate_triggers):
            return "certificate"
        if any(t.lower() in text_lower for t in transcript_keywords):
            return "transcript"
        if any(t.lower() in text_lower for t in marksheet_keywords):
            return "marksheet"
        if any(t.lower() in text_lower for t in certificate_keywords):
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
    async def gemini_generate_with_retry(prompt: str, schema, images: list = None, retries: int = 3):
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
        """Processes Marksheet OCR using Gemini Vision to produce structured JSON."""
        prompt = f"""
You are an academic document parser. Extract ALL subjects/courses from this marksheet page.

RULES:
1. Extract EVERY course listed on the page - do NOT skip or truncate.
2. For credit_points: use the numeric value if present, use '---' for non-credit or missing.
3. For registration_no and name: extract from header if visible, else leave empty string.
4. For gpa: extract the G.P.A. / GPA value if shown, else leave empty.
5. NEVER use 'NaN'. Use '---' for any missing numeric field.
6. Extract course codes EXACTLY as shown (e.g. 'Agron.3.4', 'ABM 501', 'HORT. 6.6').

OCR TEXT:
{ocr_text}

Return ONLY valid JSON in this exact format:
{{
  "registration_no": "...",
  "name": "...",
  "gpa": "...",
  "subjects": [
    {{ "code": "...", "title": "...", "credit_points": "..." }}
  ]
}}
"""

        if GEMINI_API_KEY:
            try:
                images = image_data if image_data else None
                result = await ProcessingService.gemini_generate_with_retry(prompt, MarkSheetData, images=images)
                logger.info(f"Gemini Marksheet Extraction successful (Vision: {bool(images)}).")
                return result
            except Exception as e:
                logger.warning(f"Gemini Marksheet Extraction failed: {e}. Falling back to Cerebras.")

        if CEREBRAS_API_KEY:
            cerebras_result = await ProcessingService.generate_with_cerebras(prompt)
            if cerebras_result:
                return cerebras_result

        raise ValueError("AI Extraction failed.")

    @staticmethod
    async def extract_transcript_with_ai(image_data, ocr_text: str):
        """Processes Transcript OCR text using Gemini Vision. Handles multi-page documents page-by-page."""
        prompt = f"""
You are an expert academic transcript parser. Your job is to extract EVERY piece of data from this single page of a transcript.

#### CRITICAL SEMESTER-TO-YEAR MAPPING RULE ####
You MUST map semester numbers to year labels using this FIXED mapping:
- 1st Semester, 2nd Semester  -> "FIRST YEAR"
- 3rd Semester, 4th Semester  -> "SECOND YEAR"
- 5th Semester, 6th Semester  -> "THIRD YEAR"
- 7th Semester, 8th Semester  -> "FOURTH YEAR"
- 1st Sem (PG), 2nd Sem (PG)  -> "FIRST YEAR" (for post-grad records)
- 3rd Sem (PG), 4th Sem (PG)  -> "SECOND YEAR" (for post-grad records)

Semester ordinal words: "First"=1, "Second"=2, "Third"=3, "Fourth"=4, "Fifth"=5, "Sixth"=6, "Seventh"=7, "Eighth"=8.

#### EXTRACTION RULES ####
1. Extract student header: registration_no, name, degree, admission_year, completion_year, ogpa, result, class_division.
2. Extract EVERY course listed on this page - do NOT skip or truncate any.
3. For credit_points: extract the numeric value shown. Use '---' for non-credit courses (marked with @, grade S, or no points).
4. For GPA/CGPA: look for "G.P.A." or "GPA" and "C.G.P.A." or "CGPA" near the semester footer.
5. VERTICAL ROW OFFSET: If the semester header line ends with a number (e.g. "THIRD SEMESTER 7.8"), that number is the credit_points for the FIRST course in that semester. Each subsequent course's credit_points comes from the PREVIOUS course's line.
6. ROMAN NUMERALS: Convert OCR errors - "1" at end -> "I", "ll"/"11" -> "II".
7. COURSE CODES: Remove internal spaces ("Agron. 3.4" -> "Agron.3.4").
8. If a field is not present on this page, use empty string "".

FORMATTING:
- year: MUST be one of: "FIRST YEAR", "SECOND YEAR", "THIRD YEAR", "FOURTH YEAR"
- semester: MUST be one of: "FIRST SEMESTER", "SECOND SEMESTER", "THIRD SEMESTER", "FOURTH SEMESTER", "FIFTH SEMESTER", "SIXTH SEMESTER", "SEVENTH SEMESTER", "EIGHTH SEMESTER"

OCR TEXT FROM THIS PAGE:
{ocr_text}

Return ONLY valid JSON:
{{
  "registration_no": "",
  "name": "",
  "degree": "",
  "admission_year": "",
  "completion_year": "",
  "ogpa": "",
  "result": "",
  "class_division": "",
  "years": [
    {{
      "year": "FIRST YEAR",
      "semesters": [
        {{
          "semester": "FIRST SEMESTER",
          "gpa": "",
          "cgpa": "",
          "courses": [
            {{ "course_number": "", "title": "", "credit_points": "" }}
          ]
        }}
      ]
    }}
  ]
}}
"""
        try:
            images = image_data if image_data else None
            result = await ProcessingService.gemini_generate_with_retry(prompt, TranscriptData, images=images)
            logger.info(f"Gemini Transcript Extraction successful (Vision: {bool(images)}).")
            return result
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
            # If OCR failed or is poor quality, use Vision
            # (Certificates are usually 1 page, image_data could be bytes or list)
            use_vision = "OCR Failed" in ocr_text or len(ocr_text) < 50
            images = image_data if isinstance(image_data, list) else [image_data]
            images = images if use_vision else None

            result = await ProcessingService.gemini_generate_with_retry(prompt, CertificateData, images=images)
            if result.get("ogpa"):
                # Clean up OGPA to remove any scale like "/ 10.00"
                result["ogpa"] = str(result["ogpa"]).split('/')[0].strip()
            return result
        except Exception as e:
            logger.error(f"Certificate Extraction failed: {e}")
            raise e
