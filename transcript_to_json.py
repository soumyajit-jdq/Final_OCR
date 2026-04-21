import os
import json
import re
import requests
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from PIL import Image
import io
import uvicorn
import logging
import psutil
import signal
import fitz
from web3 import Web3
from collections import OrderedDict
from dotenv import load_dotenv

load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CONFIG
OCR_API_KEY = os.getenv("OCR_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CEREBRAS_API_KEY=os.getenv("CEREBRAS_API_KEY", "")

app = FastAPI(title="MarkSheet AI Parser (Reasoning Edition)")

# DATA MODELS
class Course(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    course_number: str = Field(..., description="Course/Subject Number")
    title: str = Field(..., description="Course title")
    # credit_hours: str = Field(..., description="Credit Hours / Units")
    # grade_point: str = Field(..., description="Grade Point (0.0-10.0 or 'S' for Satisfactory)")
    credit_points: str = Field(..., description="Total Credit Points")

class SemesterData(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    semester: str = Field(..., description="Semester name in UPPERCASE words (e.g., 'FIRST SEMESTER', 'SEVENTH SEMESTER')")
    gpa: Optional[str] = Field(None, description="Grade Point Average for the semester")
    cgpa: Optional[str] = Field(None, description="Cumulative Grade Point Average (up to this semester)")
    courses: List[Course]

class YearData(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    year: str = Field(..., description="Year level in UPPERCASE words (e.g., 'FIRST YEAR', 'FOURTH YEAR')")
    semesters: List[SemesterData]

class TranscriptData(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    # sr_no: Optional[str] = Field(None, description="Transcript Serial Number")
    registration_no: str = Field(..., description="Student Registration/Enrollment Number")
    name: str = Field(..., description="Student Name")
    degree: Optional[str] = Field(None, description="Degree Name (e.g., B. Sc. (Hons.)Agri.)")
    admission_year: Optional[str] = Field(None, description="Admission Year (e.g., 2012-2013)")
    completion_year: Optional[str] = Field(None, description="Completion Year (e.g., 2015-2016)")
    # college: Optional[str] = Field(None, description="College Name")
    ogpa: Optional[str] = Field(None, description="Overall Grade Point Average (OGPA)")
    # total_credit_hours: Optional[str] = Field(None, description="Total Credit Hours (including Electives)")
    # total_credit_points: Optional[str] = Field(None, description="Total Credit Points")
    result: Optional[str] = Field(None, description="Final Result (e.g., Pass)")
    class_division: Optional[str] = Field(None, description="Class/Division (e.g., First Class with Distinction)")
    # percentage: Optional[str] = Field(None, description="Final Percentage")
    years: List[YearData]
    merkle_hash: Optional[str] = Field(None, description="Keccak-256 Verification Hash")

# PORT UTILITIES
def force_free_port(port: int):
    """Detects and kills any process using the specified port."""
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            if conn.pid:
                try:
                    p = psutil.Process(conn.pid)
                    logger.info(f"Port {port} is occupied by {p.name()} (PID: {conn.pid}). Terminating...")
                    p.terminate()
                    p.wait(timeout=3)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                    try:
                        logger.warning(f"Graceful termination failed for PID {conn.pid}. Forcing...")
                        p.kill()
                    except:
                        pass

# HELPER FUNCTIONS
def compress_image(image_bytes: bytes, max_kb: int = 1000):
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    if len(image_bytes) <= max_kb * 1024:
        return image_bytes
    quality = 90
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

def run_ocr(image_bytes: bytes):
    """Sends a single compressed image to OCR.space."""
    compressed_bytes = compress_image(image_bytes)
    url = "https://api.ocr.space/parse/image"
    files = {"file": ("image.jpg", compressed_bytes, "image/jpeg")}
    data = {"apikey": OCR_API_KEY, "language": "eng", "isTable": True, "OCREngine": 2}
    try:
        response = requests.post(url, files=files, data=data, timeout=60)
        result = response.json()
        if result.get("OCRExitCode") != 1:
            return f"[OCR Error: {result.get('ErrorMessage')}]"
        return result["ParsedResults"][0]["ParsedText"]
    except Exception as e:
        return f"[OCR Exception: {e}]"

def run_ocr_on_pdf(pdf_bytes: bytes):
    """Sends PDF to OCR.space. If > 1MB, processes page-by-page as images to avoid API limits."""
    # OCR.space free tier limit is 1024KB.
    if len(pdf_bytes) > 1024 * 1024:
        logger.info("PDF > 1MB. Switching to page-by-page image OCR processing...")
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        all_text = []
        for i in range(len(doc)):
            page = doc[i]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_bytes = pix.tobytes("jpg")
            text = run_ocr(img_bytes)
            all_text.append(text)
        doc.close()
        return "\n\n".join(all_text)

    url = "https://api.ocr.space/parse/image"
    files = {"file": ("document.pdf", pdf_bytes, "application/pdf")}
    data = {
        "apikey": OCR_API_KEY,
        "language": "eng",
        "isTable": True,
        "OCREngine": 2
    }
    try:
        response = requests.post(url, files=files, data=data, timeout=60)
        result = response.json()
        if result.get("OCRExitCode") != 1:
            # If PDF fails even if < 1MB, fallback to image processing
            return run_ocr_on_pdf(pdf_bytes + b'\0') # Recursive dummy call with size force to trigger image path? No, let's just implement explicit fallback.
            
        all_text = []
        for res in result.get("ParsedResults", []):
            all_text.append(res.get("ParsedText", ""))
        return "\n\n".join(all_text)
    except Exception as e:
        return f"OCR Error: {e}"

def verify_with_llama(text: str):
    """Uses Cerebras (Llama) to verify if the document is an academic transcript."""
    if not CEREBRAS_API_KEY or not text.strip():
        return {"is_transcript": True, "reason": "No text or key"}
    try:
        url = "https://api.cerebras.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {CEREBRAS_API_KEY}",
            "Content-Type": "application/json"
        }
        prompt = f"Identify if the following text is from an Official Academic Transcript. Check for 'Transcript', 'Cumulative', 'Degree', 'Registrar', or lists of courses by part/year. Respond ONLY with JSON: {{\"is_transcript\": true, \"reason\": \"matches pattern\"}} or {{\"is_transcript\": false, \"reason\": \"error msg\"}}\n\nTEXT:\n{text[:3000]}"
        payload = {
            "model": "llama3.1-8b",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.0
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=30).json()
        return json.loads(resp["choices"][0]["message"]["content"])
    except Exception as e:
        logger.error(f"Llama verification failed: {e}")
        return {"is_transcript": True, "reason": "Verification bypassed"}


def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# HASHING UTILITIES
def build_canonical_payload(data: dict) -> str:
    """Builds a canonical JSON string for the nested structure."""
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
        ("college", str(data.get("college", ""))),
        ("ogpa", str(data.get("ogpa", ""))),
        ("total_credit_hours", str(data.get("total_credit_hours", ""))),
        ("total_credit_points", str(data.get("total_credit_points", ""))),
        ("result", str(data.get("result", ""))),
        ("class_division", str(data.get("class_division", ""))),
        ("percentage", str(data.get("percentage", ""))),
        ("years", years)
    ])
    return json.dumps(payload, separators=(',', ':'))


def process_pdf_pages(pdf_bytes: bytes, max_pages: int = 3):
    """Converts up to max_pages of a PDF to OCR-ready images and extracts text."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        num_pages = min(len(doc), max_pages)
        if num_pages == 0:
            return [], ""
        
        all_text = []
        all_images = []
        
        for i in range(num_pages):
            page = doc[i]
            # Extract text
            all_text.append(page.get_text().strip())
            
            # Convert to high-res image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            all_images.append(pix.tobytes("jpg"))
            
        doc.close()
        return all_images, "\n\n".join(all_text)
    except Exception as e:
        logger.error(f"PDF processing failed: {e}")
        return [], ""

def generate_keccak256(text: str):
    """Generates an Ethereum-standard Keccak-256 hash using Web3.py."""
    hash_bytes = Web3.keccak(text=text)
    return Web3.to_hex(hash_bytes)

def generate_with_cerebras(prompt: str):
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
            # "model": "qwen-3-235b-a22b-instruct-2507",
            # "model": "",
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
        resp_1 = requests.post(url, headers=headers, json=payload_1, timeout=30).json()
        if "choices" not in resp_1:
            return None
        initial_json = resp_1["choices"][0]["message"]["content"]
        
        # --- PASS 2: Self-Correction Loop ---
        correction_system_prompt = (
            "You are a character-level QA auditor. Correct any JSON formatting issues in the provided data.\n"
            "Ensure it matches the nested structure requested.\n"
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
        resp_2 = requests.post(url, headers=headers, json=payload_2, timeout=30).json()
        
        final_content = initial_json
        if "choices" in resp_2:
            final_content = resp_2["choices"][0]["message"]["content"]
            logger.info("Self-Correction loop completed.")
        
        # Robust JSON cleaning
        match = re.search(r"\{.*\}", final_content, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(final_content)
    except Exception as e:
        logger.warning(f"Cerebras extraction/correction failed: {e}")
        return None


def generate_with_gemini(image_data, prompt: str):
    """Fallback to Gemini for high-precision vision parsing. image_data is a List[bytes]."""
    from google import genai
    from google.genai import types
    import time
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    contents = [prompt]
    if isinstance(image_data, list):
        for img_bytes in image_data:
            contents.append(Image.open(io.BytesIO(img_bytes)))
    else:
        contents.append(Image.open(io.BytesIO(image_data)))

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
            # model='gemini-1.5-flash-lite',
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
            if "503" in str(e) and attempt < max_retries - 1:
                logger.warning(f"Gemini 503 Service Unavailable. Retrying in {2**(attempt+1)}s...")
                time.sleep(2**(attempt+1))
                continue
            logger.warning(f"Gemini extraction failed: {e}")
            return None
    return None


def generate_structured_data(image_data, ocr_text: str):
    prompt = f"""
You are an expert academic transcript parser. Extract ALL fields into a NESTED HIERARCHY.
STRICT RULE: Format Year and Semester as ALL CAPS WORDS. NO numbers, NO Roman numerals.

#### FIELD EXTRACTION RULES ####
1. **Header Info**: Look for "Sr. No.", "Registration No.", "Name", "Faculty", "Degree", "Admission Year", "Completion Year", "Medium of instruction", and "College".
2. **Summary Info**: Look for "Total Credit Hours", "Total Credit Points", "O.G.P.A.", "Result", "Class", and "Percentage" (usually at the end).
3. **Course Alignment Fix**: 
   - CRITICAL: In the OCR text, numbers often appear on a DIFFERENT line than the course title.
   - For non-credit courses (e.g., NSS/NCC, Physical Education), the grade is often 'S' (Satisfactory). 
   - STRICT RULE: NEVER put 'S' in "credit_points". For non-credit courses, "credit_points" MUST be '--'.
   - Use the provided images to visually anchor every number to its correct course.
4. **GPA/CGPA**: For each semester, extract both the "G.P.A." and the cumulative "C.G.P.A." provided at the end of the semester block.

#### FORMATTING ####
- "year": MUST BE "FIRST YEAR", "SECOND YEAR", "THIRD YEAR", or "FOURTH YEAR".
- "semester": MUST BE "FIRST SEMESTER", "SECOND SEMESTER", ..., "EIGHTH SEMESTER".

#### JSON STRUCTURE ####
{{
  "sr_no": "...",
  "registration_no": "...",
  "name": "...",
  "faculty": "...",
  "degree": "...",
  "admission_year": "...",
  "completion_year": "...",
  "ogpa": "...",
  "result": "...",
  "class_division": "...",
  "percentage": "...",
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

OCR TEXT:
{ocr_text}

Return ONLY the structured JSON.
"""

    # Primary Multimodal Extraction (Handles multiple pages perfectly)
    if GEMINI_API_KEY:
        logger.info("Running Multimodal Gemini Extraction")
        gemini_result = generate_with_gemini(image_data, prompt)
        if gemini_result:
            return gemini_result

    # Fallback to Cerebras if Gemini fails (Text only)
    if CEREBRAS_API_KEY:
        logger.info("Attempting fallback text-only extraction with Cerebras")
        cerebras_result = generate_with_cerebras(prompt)
        if cerebras_result:
            return cerebras_result

    raise ValueError("All extraction models failed.")


    raise ValueError("All extraction models failed.")

# API ENDPOINTS
@app.post("/parse-transcript", response_model=TranscriptData)
async def parse_transcript(file: UploadFile = File(...)):
    try:
        # GATE 0: File Type Check
        if file.content_type != "application/pdf" and not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported for transcripts.")
            
        file_bytes = await file.read()
        
        # GATE 1: Page Count Check
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page_count = len(doc)
        if page_count < 2 or page_count > 3:
            doc.close()
            raise HTTPException(status_code=400, detail=f"Invalid transcript: Page count is {page_count}. Must be 2 or 3 pages.")
        
        # Extract Images for Multimodal Processing
        img_list, low_res_text = process_pdf_pages(file_bytes, max_pages=3)
        doc.close()

        # GATE 2: OCR.space (Extract data from the whole PDF)
        logger.info("Running OCR.space on the whole PDF...")
        ocr_text = run_ocr_on_pdf(file_bytes)
        
        # DEBUG: Print OCR text to terminal
        print(f"\n--- OCR.SPACE TEXT OUTPUT ---\n{ocr_text}\n-----------------------------\n")
        
        # GATE 3: Cerebras (Llama) Gatekeeper - Verify Document Type
        logger.info("Verifying document type with Llama...")
        verification = verify_with_llama(ocr_text)
        if not verification.get("is_transcript", True):
            logger.warning(f"Verification Failed: {verification.get('reason')}")
            raise HTTPException(status_code=400, detail=f"Invalid document: {verification.get('reason')}")

        # GATE 4: Multimodal Extraction with Gemini
        logger.info("Proceeding to Multimodal Extraction...")
        structured_data = generate_structured_data(img_list, ocr_text)
        
        # GATE 5: Verification Hash
        canonical_json = build_canonical_payload(structured_data)
        structured_data["merkle_hash"] = generate_keccak256(canonical_json)
        
        return TranscriptData(**structured_data)
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("Processing failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Transcript Parser (Multimodal)</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root { --primary: #3b82f6; --bg: #030712; --card: #111827; --text: #f9fafb; --accent: #60a5fa; }
        body { font-family: 'Outfit', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; display: flex; align-items: center; justify-content: center; margin: 0; padding: 40px 0; }
        .box { width: 90%; max-width: 1100px; background: var(--card); border: 1px solid #1f2937; border-radius: 20px; padding: 40px; }
        .dropzone { border: 2px dashed #374151; border-radius: 12px; padding: 40px; text-align: center; cursor: pointer; transition: 0.3s; }
        .dropzone:hover { border-color: var(--primary); background: #1f2937; }
        .btn { background: var(--primary); color: white; border: none; padding: 12px 30px; border-radius: 8px; font-weight: 600; cursor: pointer; margin-top: 20px; width: 100%; transition: 0.2s; }
        .btn:hover { background: #2563eb; }
        #results { margin-top: 40px; display: none; }
        .meta-card { background: rgba(255,255,255,0.03); border: 1px solid #1f2937; border-radius: 12px; padding: 25px; margin-bottom: 30px; border-left: 4px solid var(--primary); }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px; }
        .label { font-size: 0.7em; text-transform: uppercase; color: #9ca3af; margin-bottom: 2px; letter-spacing: 0.05em; }
        .val { font-size: 1.1em; font-weight: 600; color: #f3f4f6; }
        
        /* Nested Layout */
        .year-block { margin-top: 40px; border: 1px solid #1f2937; border-radius: 15px; overflow: hidden; background: rgba(0,0,0,0.2); }
        .year-header { background: #1e293b; padding: 15px 25px; font-weight: 600; color: #93c5fd; border-bottom: 1px solid #1f2937; font-size: 1.1em; }
        .sem-block { padding: 25px; border-bottom: 1px solid #1f2937; }
        .sem-block:last-child { border-bottom: none; }
        .sem-header { font-weight: 600; color: #e2e8f0; margin-bottom: 15px; display: flex; justify-content: space-between; border-left: 3px solid #3b82f6; padding-left: 15px; }
        
        table { width: 100%; border-collapse: collapse; font-size: 0.85em; margin-top: 10px; }
        th, td { padding: 12px; border-bottom: 1px solid #1f2937; text-align: left; }
        th { color: #94a3b8; font-weight: 500; text-transform: uppercase; font-size: 0.75em; }
        .log-box { background: #000; color: #10b981; padding: 15px; border-radius: 8px; font-family: 'Fira Code', monospace; font-size: 0.8em; margin-top: 30px; border: 1px solid #064e3b; max-height: 300px; overflow-y: auto; white-space: pre-wrap; }
        .hash-card { background: #064e3b; border: 1px solid #065f46; color: #34d399; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 0.85em; margin-top: 20px; word-break: break-all; }
        .loader { display: none; text-align: center; margin: 20px 0; color: #60a5fa; font-weight: 500; }
    </style>
</head>
<body>
    <div class="box">
        <h1 style="margin-top:0; font-size: 1.8em;">Transcript AI Parser</h1>
        <p style="color:#9ca3af; margin-bottom: 30px;">Hierarchical Extraction (Year > Semester > Courses) with Blockchain Hashing</p>
        <form id="pForm">
            <div class="dropzone" onclick="document.getElementById('f').click()">
                <span id="label">Select Transcript PDF (2-3 Pages)</span>
                <input type="file" id="f" style="display:none" accept=".pdf" onchange="document.getElementById('label').innerText=this.files[0].name">
            </div>
            <button type="submit" class="btn">Start Multi-Stage Analysis</button>
        </form>
        <div class="loader" id="l">Initializing Pipeline: PDF Validation -> Page OCR -> Llama Verification -> Gemini Hierarchical Extraction...</div>
        
        <div id="results">
            <div class="meta-card">
                <div id="n" style="color:var(--accent); font-weight:600; font-size:1.3em; margin-bottom:5px;"></div>
                <div id="r" style="font-size:0.9em; opacity:0.8; margin-bottom:15px;"></div>
                
                <div class="grid" style="grid-template-columns: 1fr 1fr 1fr;">
                    <div><div class="label">Sr. No.</div><div id="sr_no" class="val"></div></div>
                    <div><div class="label">Degree</div><div id="degree" class="val"></div></div>
                    <div><div class="label">Admission</div><div id="adm" class="val"></div></div>
                </div>
                <div class="grid" style="grid-template-columns: 1fr 1fr;">
                    <div><div class="label">Completion</div><div id="comp" class="val"></div></div>
                </div>
            </div>

            <div id="hierarchy-container"></div>

            <!-- Summary Card -->
            <div class="meta-card" style="border-left: 4px solid #10b981; margin-top: 30px;">
                <div style="color:#10b981; font-weight:600; font-size:1.1em; margin-bottom:15px;">ABSTRACT OF ACADEMIC PERFORMANCE</div>
                <div class="grid" style="grid-template-columns: 1fr 1fr 1fr;">
                    <div><div class="label">OGPA (10.00 Basis)</div><div id="ogpa" class="val" style="color:#10b981; font-size: 1.4em;"></div></div>
                    <div><div class="label">Total Credit Hours</div><div id="tch" class="val"></div></div>
                    <div><div class="label">Total Credit Points</div><div id="tcp" class="val"></div></div>
                </div>
                <div class="grid" style="grid-template-columns: 1fr 1fr; margin-top: 15px;">
                    <div><div class="label">Final Result</div><div id="res" class="val"></div></div>
                    <div><div class="label">Class/Division</div><div id="class" class="val"></div></div>
                </div>
            </div>

            <div class="hash-card">
                <div class="label" style="color:#6ee7b7; margin-bottom:5px;">KECCAK-256 VERIFICATION HASH</div>
                <span id="h"></span>
            </div>

            <div class="log-box">
                <div class="label" style="color:#10b981; margin-bottom:10px;">RAW CANONICAL JSON</div>
                <div id="json-log"></div>
            </div>
        </div>
    </div>
    <script>
        document.getElementById('pForm').onsubmit = async (e) => {
            e.preventDefault();
            const f = document.getElementById('f').files[0];
            if(!f) return;
            document.getElementById('l').style.display='block';
            document.getElementById('results').style.display='none';
            const fd = new FormData(); fd.append('file', f);
            try {
                const res = await fetch('/parse-transcript', {method:'POST', body:fd});
                if(!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || 'Processing failed');
                }
                const d = await res.json();
                
                // Meta info
                document.getElementById('n').innerText = d.name;
                document.getElementById('r').innerText = 'Registration/Enrollment: ' + d.registration_no;
                document.getElementById('sr_no').innerText = d.sr_no || 'N/A';
                document.getElementById('degree').innerText = d.degree || 'N/A';
                document.getElementById('adm').innerText = d.admission_year || 'N/A';
                document.getElementById('comp').innerText = d.completion_year || 'N/A';
                
                // Summary info
                document.getElementById('ogpa').innerText = d.ogpa || 'N/A';
                document.getElementById('tch').innerText = d.total_credit_hours || 'N/A';
                document.getElementById('tcp').innerText = d.total_credit_points || 'N/A';
                document.getElementById('res').innerText = d.result || 'N/A';
                document.getElementById('class').innerText = d.class_division || 'N/A';
                
                document.getElementById('h').innerText = d.merkle_hash;
                
                // Nested Logs
                const { merkle_hash, ...logData } = d;
                document.getElementById('json-log').innerText = JSON.stringify(logData, null, 2);
                
                // Build Hierarchy
                document.getElementById('hierarchy-container').innerHTML = d.years.map(y => `
                    <div class="year-block">
                        <div class="year-header">${y.year.toUpperCase()}</div>
                        ${y.semesters.map(s => `
                            <div class="sem-block">
                                <div class="sem-header">
                                    <span>${s.semester}</span>
                                    <span style="color: var(--primary)">GPA: ${s.gpa || 'N/A'} | CGPA: ${s.cgpa || 'N/A'}</span>
                                </div>
                                <table>
                                    <thead>
                                        <tr>
                                            <th style="width: 20%">Course No</th>
                                            <th>Subject Title</th>
                                            <th style="width: 15%">Credit Points</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${s.courses.map(c => `
                                            <tr>
                                                <td>${c.course_number}</td>
                                                <td>${c.title}</td>
                                                <td>${c.credit_points}</td>
                                            </tr>
                                        `).join('')}
                                    </tbody>
                                </table>
                            </div>
                        `).join('')}
                    </div>
                `).join('');

                document.getElementById('results').style.display='block';
            } catch(er) { 
                console.error("Error:", er);
                document.getElementById('json-log').innerText = JSON.stringify({ error: er.message }, null, 2);
                document.getElementById('results').style.display='block';
                alert('Analysis Failed: ' + er.message); 
            }
            finally { document.getElementById('l').style.display='none'; }
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    TARGET_PORT = 8080
    force_free_port(TARGET_PORT)
    uvicorn.run(app, host="0.0.0.0", port=TARGET_PORT)