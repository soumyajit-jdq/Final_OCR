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
from Crypto.Hash import keccak
from dotenv import load_dotenv

load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CONFIG
OCR_API_KEY = os.getenv("OCR_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

app = FastAPI(title="MarkSheet AI Parser (Reasoning Edition)")

# DATA MODELS
class Subject(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    code: str = Field(..., description="Course code")
    title: str = Field(..., description="Course title")
    credits: str = Field(..., description="Credit hours")
    grade: str = Field(..., description="Grade awarded")
    marks_obtained: Optional[str] = None
    marks_total: Optional[str] = None

class MarkSheetData(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    name: str = Field(..., description="Student Name")
    university: Optional[str] = Field(None, description="University Name")
    college: Optional[str] = Field(None, description="College/Institute Name")
    registration_no: str = Field(..., description="Student Registration/Enrollment Number")
    seat_no: Optional[str] = Field(None, description="Seat Number")
    semester: Optional[str] = Field(None, description="Semester (e.g. Semester I)")
    academic_year: Optional[str] = Field(None, description="Academic Year")
    subjects: List[Subject]
    gpa: Optional[str] = Field(None, description="Grade Point Average")
    result_status: Optional[str] = Field(None, description="Final Result Status")
    date_of_issue: Optional[str] = Field(None, description="Date of Issue")
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
    compressed_bytes = compress_image(image_bytes)
    url = "https://api.ocr.space/parse/image"
    files = {"file": ("image.jpg", compressed_bytes, "image/jpeg")}
    data = {"apikey": OCR_API_KEY, "language": "eng", "isTable": True, "OCREngine": 2}
    try:
        response = requests.post(url, files=files, data=data, timeout=60)
        result = response.json()
        if result.get("OCRExitCode") != 1:
            return f"OCR Failed: {result.get('ErrorMessage')}"
        return result["ParsedResults"][0]["ParsedText"]
    except Exception as e:
        return f"OCR Error: {e}"

def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# HASHING UTILITIES
def normalize_for_hash(data: dict):
    """Creates a deterministic string for hashing."""
    parts = []
    # Core Metadata
    parts.append(str(data.get("university", "")).strip())
    parts.append(str(data.get("college", "")).strip())
    parts.append(str(data.get("name", "")).strip())
    parts.append(str(data.get("registration_no", "")).strip())
    parts.append(str(data.get("semester", "")).strip())
    
    # Subjects (Sorted by Code for determinism)
    subs = data.get("subjects", [])
    sorted_subs = sorted(subs, key=lambda x: x.get("code", ""))
    for s in sorted_subs:
        parts.append(f"{s.get('code')}{s.get('grade')}{s.get('credits')}")
    
    parts.append(str(data.get("gpa", "")).strip())
    return "|".join(parts).lower()

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
    k = keccak.new(digest_bits=256)
    k.update(text.encode('utf-8'))
    return k.hexdigest()

def generate_with_gemini(image_data, prompt: str):
    """Fallback to Gemini for high-precision vision parsing. image_data can be bytes or List[bytes]."""
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        contents = [prompt]
        if isinstance(image_data, list):
            for img_bytes in image_data:
                contents.append(Image.open(io.BytesIO(img_bytes)))
        else:
            contents.append(Image.open(io.BytesIO(image_data)))
        
        response = client.models.generate_content(
            # model='gemini-3.1-flash-lite-preview',
            model='gemini-2.5-flash-pro',
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MarkSheetData,
                temperature=0.0
            )
        )
        return json.loads(response.text)
    except Exception as e:
        logger.warning(f"Gemini extraction failed: {e}")
        return None

def generate_structured_data(image_data, ocr_text: str):
    # If it's a list (multi-page PDF), we use the first page as the primary for fallback models
    # but Gemini will get all pages.
    primary_image_bytes = image_data[0] if isinstance(image_data, list) else image_data
    base64_image = encode_image(primary_image_bytes)
    
    prompt = f"""
You are an expert marksheet parser. Extract ALL details from the provided OCR text and image with 100% literal precision.
The document may contain trilingual text (English, Gujarati, Hindi). Extract exactly as visible.

OCR TEXT:
{ocr_text}

JSON FORMAT:
{{
  "name": "Full Name",
  "university": "University Name",
  "college": "College Name",
  "registration_no": "Enrollment/Reg No",
  "seat_no": "Seat No (if any)",
  "semester": "Semester",
  "academic_year": "Year",
  "subjects": [
    {{
      "code": "Code", "title": "Subject Title", "credits": "CR", "grade": "GR",
      "marks_obtained": "Marks (if any)", "marks_total": "Total (if any)"
    }}
  ],
  "gpa": "GPA/SGPA/CGPA",
  "result_status": "Pass/Fail/Result Status",
  "date_of_issue": "Date"
}}
Return ONLY the JSON.
"""

    # Try Gemini First (Handles multiple images natively)
    if GEMINI_API_KEY:
        logger.info("Attempting primary extraction with Gemini...")
        gemini_result = generate_with_gemini(image_data, prompt)
        if gemini_result:
            return gemini_result

    # Using Nvidia Vision-Language model as secondary (requested by user)
    models_to_try = [
        "nvidia/nemotron-nano-12b-v2-vl:free",
        "google/gemma-4-31b-it:free",
        "liquid/lfm-2.5-1.2b-thinking:free",
        "qwen/qwen3-next-80b-a3b-instruct:free"
    ]
    
    for model_id in models_to_try:
        try:
            logger.info(f"Attempting reasoning-enabled extraction with {model_id}...")
            payload = {
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                "reasoning": {"enabled": True}
            }
            
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=90)
            res_json = response.json()
            
            if "choices" not in res_json:
                logger.warning(f"Model {model_id} error: {res_json.get('error')}")
                continue
                
            choice = res_json["choices"][0]
            if "reasoning_details" in choice["message"]:
                logger.debug(f"AI Reasoning: {choice['message']['reasoning_details']}")
                
            response_text = choice["message"]["content"]
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if match:
                logger.info(f"Successfully used {model_id}!")
                return json.loads(match.group())
                
        except Exception as e:
            logger.warning(f"Failed with {model_id}: {e}")
            continue
            
    raise ValueError("All reasoning-enabled models failed.")

# API ENDPOINTS
@app.post("/parse-marksheet", response_model=MarkSheetData)
async def parse_marksheet(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        ocr_text = ""
        processing_image = file_bytes
        
        # Handle PDF vs Image
        if file.content_type == "application/pdf" or file.filename.lower().endswith(".pdf"):
            logger.info("Processing PDF input (extracting up to 3 pages)...")
            img_list, raw_text = process_pdf_pages(file_bytes)
            if img_list:
                processing_image = img_list # Now it's a list
                ocr_text = raw_text
        
        # Run OCR update if text is sparse (scanned documents)
        # Use primary image if it's a list
        ocr_source_image = processing_image[0] if isinstance(processing_image, list) else processing_image
        if len(ocr_text) < 50:
            logger.info("Running OCR.space on document...")
            ocr_text = run_ocr(ocr_source_image)
            
        try:
            print(f"\n--- OCR TEXT ---\n{ocr_text}\n---------------\n")
        except UnicodeEncodeError:
            logger.info("OCR Text contains non-ASCII characters; skipping direct print to terminal.")
        
        structured_data = generate_structured_data(processing_image, ocr_text)
        
        # Calculate Verification Hash
        norm_text = normalize_for_hash(structured_data)
        structured_data["merkle_hash"] = generate_keccak256(norm_text)
        
        return MarkSheetData(**structured_data)
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
    <title>AI MarkSheet Reasoning Parser</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root { --primary: #8b5cf6; --bg: #030712; --card: #111827; --text: #f9fafb; }
        body { font-family: 'Outfit', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; display: flex; align-items: center; justify-content: center; margin: 0; }
        .box { width: 90%; max-width: 900px; background: var(--card); border: 1px solid #1f2937; border-radius: 20px; padding: 30px; }
        .dropzone { border: 2px dashed #374151; border-radius: 12px; padding: 40px; text-align: center; cursor: pointer; transition: 0.3s; }
        .dropzone:hover { border-color: var(--primary); background: #1f2937; }
        .btn { background: var(--primary); color: white; border: none; padding: 12px 30px; border-radius: 8px; font-weight: 600; cursor: pointer; margin-top: 20px; width: 100%; transition: 0.2s; }
        .btn:hover { background: #7c3aed; }
        #results { margin-top: 30px; display: none; }
        .meta-card { background: rgba(255,255,255,0.03); border: 1px solid #1f2937; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px; }
        .label { font-size: 0.7em; text-transform: uppercase; color: #9ca3af; margin-bottom: 2px; }
        .val { font-size: 0.95em; font-weight: 600; color: #f3f4f6; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 0.85em; }
        th, td { padding: 10px; border-bottom: 1px solid #1f2937; text-align: left; }
        th { color: #a78bfa; font-weight: 600; }
        .hash-box { background: #064e3b; color: #34d399; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 0.8em; margin-top: 20px; border: 1px solid #065f46; word-break: break-all; }
        .log-box { background: #000; color: #00ff00; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 0.8em; margin-top: 20px; border: 1px solid #333; height: 150px; overflow-y: auto; white-space: pre-wrap; }
        .loader { display: none; text-align: center; margin: 20px 0; color: #a78bfa; }
    </style>
</head>
<body>
    <div class="box">
        <h1 style="margin-top:0;">Perfect MarkSheet Parser</h1>
        <p style="color:#9ca3af;">Deep Reasoning + Keccak-256 Verification</p>
        <form id="pForm">
            <div class="dropzone" onclick="document.getElementById('f').click()">
                <span id="label">Select MarkSheet Scan (Image/PDF)</span>
                <input type="file" id="f" style="display:none" accept="image/*,.pdf" onchange="document.getElementById('label').innerText=this.files[0].name">
            </div>
            <button type="submit" class="btn">Start Blockchain Analysis</button>
        </form>
        <div class="loader" id="l">⚡ AI is Reasoning through your document...</div>
        <div id="results">
            <div class="meta-card">
                <div id="uni" style="color:var(--primary); font-weight:600; font-size:1.1em; margin-bottom:2px;"></div>
                <div id="coll" style="font-size:0.9em; opacity:0.8; margin-bottom:15px;"></div>
                <div class="grid">
                    <div><div class="label">Student Name</div><div id="n" class="val"></div></div>
                    <div><div class="label">Reg No / Enrollment</div><div id="r" class="val"></div></div>
                    <div><div class="label">Semester</div><div id="sm" class="val"></div></div>
                    <div><div class="label">Academic Year</div><div id="ay" class="val"></div></div>
                </div>
            </div>
            <table>
                <thead><tr><th>Code</th><th>Subject</th><th>Credits</th><th>Grade</th><th>Marks</th></tr></thead>
                <tbody id="b"></tbody>
            </table>
            <div class="grid" style="margin-top:20px;">
                <div><div class="label">GPA</div><div id="g" class="val"></div></div>
                <div><div class="label">Status</div><div id="st" class="val"></div></div>
            </div>
            <div class="hash-box">
                <div class="label" style="color:#6ee7b7; margin-bottom:5px;">KECCAK-256 VERIFICATION HASH</div>
                <span id="h"></span>
            </div>
            <div class="log-box">
                <div class="label" style="color:#4ade80; margin-bottom:5px; font-family: sans-serif;">RAW JSON LOG</div>
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
                const res = await fetch('/parse-marksheet', {method:'POST', body:fd});
                const d = await res.json();
                document.getElementById('uni').innerText = d.university;
                document.getElementById('coll').innerText = d.college;
                document.getElementById('n').innerText = d.name;
                document.getElementById('r').innerText = d.registration_no;
                document.getElementById('sm').innerText = d.semester;
                document.getElementById('ay').innerText = d.academic_year;
                document.getElementById('g').innerText = d.gpa;
                document.getElementById('st').innerText = d.result_status || 'N/A';
                document.getElementById('h').innerText = d.merkle_hash;
                document.getElementById('json-log').innerText = JSON.stringify(d, null, 2);
                
                document.getElementById('b').innerHTML = d.subjects.map(s => `
                    <tr>
                        <td>${s.code || '-'}</td>
                        <td>${s.title || '-'}</td>
                        <td>${s.credits || '-'}</td>
                        <td>${s.grade || '-'}</td>
                        <td>${s.marks_obtained || '-'}/${s.marks_total || '-'}</td>
                    </tr>`).join('');
                document.getElementById('results').style.display='block';
                console.log("Extraction Success:", d);
            } catch(er) { 
                console.error("UI Error:", er);
                alert('Analysis Success but UI Rendering Failed: ' + er); 
            }
            finally { document.getElementById('l').style.display='none'; }
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    TARGET_PORT = 8000
    force_free_port(TARGET_PORT)
    uvicorn.run(app, host="0.0.0.0", port=TARGET_PORT)