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
        """Image compression is CPU bound, running in thread."""
        def sync_compress():
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            if len(image_bytes) <= max_kb * 1024:
                return image_bytes
            quality = 90
            buffer = io.BytesIO()
            while quality > 10:
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=quality, optimize=True)
                if len(buffer.getvalue()) <= max_kb * 1024:
                    return buffer.getvalue()
                quality -= 10
            img.thumbnail((1600, 1600))
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
            ("university", str(data.get("university", ""))),
            ("name", str(data.get("name", ""))),
            ("degree", str(data.get("degree", ""))),
            ("ogpa", str(data.get("ogpa", ""))),
            ("year", str(data.get("year", ""))),
            ("date", str(data.get("date", ""))),
            ("class_division", str(data.get("class_division", "")))
        ])
        return json.dumps(payload, separators=(',', ':'))

    @staticmethod
    async def process_pdf_pages(pdf_bytes: bytes, max_pages: int = 3):
        """PDF processing is CPU intensive, running in thread pool."""
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
        """Uses Cerebras (Llama) to categorize the document type."""
        if not CEREBRAS_API_KEY:
            return "unknown"
        try:
            url = "https://api.cerebras.ai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"}
            prompt = (
                "Identify the document type strictly based on scope:\n"
                "1. 'marksheet': A record for a SINGLE examination session, semester, or year (e.g. 'Statement of Marks', 'Grade Card', 'Evaluation Report').\n"
                "2. 'transcript': A CONSOLIDATED academic record spanning the entire degree or multiple years (e.g. 'Official Transcript', 'Consolidated Marks').\n"
                "3. 'certificate': A degree certificate, diploma, or title conferment (e.g. 'conferred upon').\n\n"
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
                # Expanded marksheet mapping for precision
                marksheet_keys = ["marksheet", "evaluation", "statement of marks", "grade card", "memo of marks", "result"]
                if any(w in content for w in marksheet_keys):
                    return "marksheet"
                if "certificate" in content:
                    return "certificate"
                if "transcript" in content:
                    return "transcript"
                return "unknown"
        except Exception as e:
            logger.error(f"Document classification failed: {e}")
            return "unknown"

    @staticmethod
    async def extract_with_ai(image_data, ocr_text: str):
        """Handles the marksheet extraction logic"""
        primary_image_bytes = image_data[0] if isinstance(image_data, list) else image_data
        base64_img = ProcessingService.encode_image(primary_image_bytes)
        
        prompt = f"""
You are an expert academic record parser. Extract details from the provided OCR text and image.
STRICT RULE: Format the output as JSON.

OCR TEXT:
{ocr_text}

JSON FORMAT:
{{
  "registration_no": "Enrollment/Reg No",
  "name": "Full Name",
  "gpa": "GPA/SGPA/CGPA",
  "subjects": [
    {{
      "code": "Code", 
      "title": "Subject Title", 
      "credit_points": "Total Credit Points ONLY"
    }}
  ]
}}
Return ONLY the JSON.
"""

        if CEREBRAS_API_KEY:
            cerebras_result = await ProcessingService.generate_with_cerebras(prompt)
            if cerebras_result:
                return cerebras_result

        if GEMINI_API_KEY:
            try:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=GEMINI_API_KEY)
                
                def bytes_to_pil(b_list):
                    if isinstance(b_list, list):
                        return [Image.open(io.BytesIO(b)) for b in b_list]
                    return Image.open(io.BytesIO(b_list))
                
                pil_images = await anyio.to_thread.run_sync(bytes_to_pil, image_data)
                contents = [prompt]
                if isinstance(pil_images, list):
                    contents.extend(pil_images)
                else:
                    contents.append(pil_images)
                
                response = await client.aio.models.generate_content(
                    model='gemini-3.1-flash-lite-preview',
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=MarkSheetData,
                        temperature=0.1
                    )
                )
                return json.loads(response.text)
            except Exception as e:
                logger.warning(f"Gemini Async failed: {e}")

        raise ValueError("AI Extraction failed.")

    @staticmethod
    async def extract_transcript_with_ai(image_data, ocr_text: str):
        """Specialized hierarchical extraction for Multi-page Transcripts."""
        prompt = f"""
You are an expert academic transcript parser. Extract ALL fields into a NESTED HIERARCHY.
STRICT RULE: Format Year and Semester as ALL CAPS WORDS.

OCR TEXT:
{ocr_text}

#### JSON STRUCTURE ####
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
      "year": "FOURTH YEAR",
      "semesters": [
        {{
            "semester": "SEVENTH SEMESTER",
            "gpa": "...",
            "cgpa": "...",
            "courses": [
              {{ 
                "course_number": "...", 
                "title": "...", 
                "credit_points": "..." 
              }}
            ]
        }}
      ]
    }}
  ]
}}

Return ONLY the structured JSON.
"""
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=GEMINI_API_KEY)
            
            def bytes_to_pil(b_list):
                if isinstance(b_list, list):
                    return [Image.open(io.BytesIO(b)) for b in b_list]
                return Image.open(io.BytesIO(b_list))
            
            pil_images = await anyio.to_thread.run_sync(bytes_to_pil, image_data)
            contents = [prompt]
            if isinstance(pil_images, list):
                contents.extend(pil_images)
            else:
                contents.append(pil_images)
            
            response = await client.aio.models.generate_content(
                model='gemini-3.1-flash-lite-preview',
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=TranscriptData,
                    temperature=0.1
                )
            )
            return json.loads(response.text)
        except Exception as e:
            logger.warning(f"Transcript Gemini Extraction failed: {e}")
            raise e

    @staticmethod
    async def extract_certificate_with_ai(image_data, ocr_text: str):
        """Specialized extraction for Academic Certificates (Degrees)."""
        prompt = f"""
You are an expert academic certificate parser. Extract specific fields from the provided document.

OCR TEXT:
{ocr_text}

#### JSON STRUCTURE ####
{{
  "certificate_no": "...",
  "no": "...",
  "university": "...",
  "name": "...",
  "degree": "...",
  "ogpa": "...",
  "year": "...",
  "date": "...",
  "class_division": "..."
}}

Return ONLY the structured JSON.
"""
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=GEMINI_API_KEY)
            
            def bytes_to_pil(b_list):
                if isinstance(b_list, list):
                    return [Image.open(io.BytesIO(b)) for b in b_list]
                return Image.open(io.BytesIO(b_list))
            
            pil_images = await anyio.to_thread.run_sync(bytes_to_pil, image_data)
            contents = [prompt]
            if isinstance(pil_images, list):
                contents.extend(pil_images)
            else:
                contents.append(pil_images)
            
            response = await client.aio.models.generate_content(
                model='gemini-3.1-flash-lite-preview',
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=CertificateData,
                    temperature=0.1
                )
            )
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Certificate AI Extraction failed: {e}")
            raise e
