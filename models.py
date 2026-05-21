from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Optional

class Subject(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    code: str = Field(..., description="Course code")
    title: str = Field(..., description="Course title")
    credits: Optional[str] = Field(None, description="Credit Hours")
    grade: Optional[str] = Field(None, description="Grade Points")
    credit_points: Optional[str] = Field(None, description="Total Credit Points")
    course_type: Optional[str] = Field(None, description="Course category or type (e.g., MAJOR-5, MDC-3, AEC-3)")
    theory_cce_min_max: Optional[str] = Field(None, description="Theory CCE Min/Max marks (e.g. 13/35)")
    theory_cce_obtained: Optional[str] = Field(None, description="Theory CCE Obtained marks")
    theory_see_min_max: Optional[str] = Field(None, description="Theory SEE Min/Max marks")
    theory_see_obtained: Optional[str] = Field(None, description="Theory SEE Obtained marks")
    practical_cce_min_max: Optional[str] = Field(None, description="Practical CCE Min/Max marks")
    practical_cce_obtained: Optional[str] = Field(None, description="Practical CCE Obtained marks")
    practical_see_min_max: Optional[str] = Field(None, description="Practical SEE Min/Max marks")
    practical_see_obtained: Optional[str] = Field(None, description="Practical SEE Obtained marks")
    total_min_max: Optional[str] = Field(None, description="Total Min/Max marks (e.g. 36/100)")
    total_obtained: Optional[str] = Field(None, description="Total Obtained marks")

class MarkSheetData(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    registration_no: str = Field(..., description="Student Registration/Enrollment Number")
    name: str = Field(..., description="Student Name")
    gpa: Optional[str] = Field(None, description="Grade Point Average")
    subjects: List[Subject]
    abc_id: Optional[str] = Field(None, description="ABC ID of the student")
    school: Optional[str] = Field(None, description="School name (e.g. SCHOOL OF SCIENCE)")
    major_course: Optional[str] = Field(None, description="Major course (e.g. Zoology)")
    examination_held_in: Optional[str] = Field(None, description="Month & Year when examination was held")
    examination_centre: Optional[str] = Field(None, description="Examination centre name/code")
    exam_type: Optional[str] = Field(None, description="Exam type (e.g. REGULAR)")
    degree_course: Optional[str] = Field(None, description="Degree course name (e.g. Bachelor of Science in Zoology(Hons))")
    semester: Optional[str] = Field(None, description="Semester (e.g. Semester - 3)")
    earned_credits: Optional[str] = Field(None, description="Earned credits for this semester")
    earned_grade_points: Optional[str] = Field(None, description="Earned grade points for this semester")
    cumulative_earned_credits: Optional[str] = Field(None, description="Cumulative earned credits")
    cumulative_earned_grade_points: Optional[str] = Field(None, description="Cumulative earned grade points")
    cgpa: Optional[str] = Field(None, description="Cumulative Grade Point Average")
    result: Optional[str] = Field(None, description="Result status (e.g. PASS, FAIL)")
    passing_cert_enrollment_no: Optional[str] = Field(None, description="Enrollment No shown on the passing certificate")
    passing_cert_month_year: Optional[str] = Field(None, description="Month & Year of Examination on passing certificate")
    passing_cert_class_obtained: Optional[str] = Field(None, description="Class obtained shown on passing certificate")
    date_of_issue: Optional[str] = Field(None, description="Date of issue of marksheet (e.g. 22/01/2026)")

class MarkSheetCollection(BaseModel):
    marksheets: List[MarkSheetData] = Field(..., description="List of all marksheets/evaluation reports found in the document")

class CertificateData(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    top_left_no: Optional[str] = Field(None, description="Top-left serial/reference number (e.g., Sr. No.)")
    certificate_no: str = Field(..., description="Certificate Number (e.g., top right number)")
    no: str = Field(..., description="Reference Number (e.g., bottom left No. suffix)")
    registration_no: Optional[str] = Field(None, description="Student Registration or Enrollment Number")
    # university: Optional[str] = Field(None, description="Issuing University Name")
    name: str = Field(..., description="Student Name")
    degree: str = Field(..., description="Degree conferred")
    branch: Optional[str] = Field(None, description="Branch/Subject of study")
    ogpa: Optional[str] = Field(None, description="Overall Grade Point Average")
    year: Optional[str] = Field(None, description="Academic Session Year")
    date: str = Field(..., description="Issue Date")
    class_division: Optional[str] = Field(None, description="Class/Division obtained")

# Hierarchical Transcript Models
class Course(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    course_number: str = Field(..., description="Course/Subject Number")
    title: str = Field(..., description="Course title")
    credits: Optional[str] = Field(None, description="Credit Hours")
    grade: Optional[str] = Field(None, description="Grade Points")
    credit_points: str = Field(..., description="Total Credit Points")

    @field_validator('course_number')
    @classmethod
    def clean_course_number(cls, v: str) -> str:
        # Remove all middle spaces from course number (e.g. "Ag. Engg. 2.1" -> "Ag.Engg.2.1")
        # Keeps internal structure but removes the 'taking space randomly' issue
        return v.replace(" ", "")

class SemesterData(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    semester: str = Field(..., description="Semester name in UPPERCASE words")
    gpa: Optional[str] = Field(None, description="GPA for the semester")
    cgpa: Optional[str] = Field(None, description="CGPA (up to this semester)")
    courses: List[Course]

class YearData(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    year: str = Field(..., description="Year level in UPPERCASE words")
    semesters: Optional[List[SemesterData]] = Field(None, description="List of semesters in this year")

class TranscriptData(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    registration_no: str = Field(..., description="Student Registration/Enrollment Number")
    name: str = Field(..., description="Student Name")
    degree: Optional[str] = Field(None, description="Degree Name")
    admission_year: Optional[str] = Field(None, description="Admission Year")
    completion_year: Optional[str] = Field(None, description="Completion Year")
    ogpa: Optional[str] = Field(None, description="Overall Grade Point Average")
    result: Optional[str] = Field(None, description="Final Result")
    class_division: Optional[str] = Field(None, description="Class/Division")
    years: Optional[List[YearData]] = Field(None, description="Academic history organized by year")
    courses: Optional[List[Course]] = Field(None, description="Flat list of courses (use if no year/semester headings exist)")

class ValidationResponse(BaseModel):
    is_valid: bool = Field(..., description="Whether the document meets quality standards")
    instruction: str = Field(..., description="User-friendly instruction for the pop-up")
    file_type: Optional[str] = None

class ProcessingResult(BaseModel):
    filename: str
    doc_type: str
    status: str
    data: Optional[dict] = None
    raw_text: Optional[str] = None
    error: Optional[str] = None
    ledger_hash: Optional[str] = None

class BulkProcessingResponse(BaseModel):
    total_files: int
    processed_files: int
    failed_files: int
    results: List[ProcessingResult]