from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional

class Subject(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    code: str = Field(..., description="Course code")
    title: str = Field(..., description="Course title")
    credit_points: str = Field(..., description="Total Credit Points")

class MarkSheetData(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    registration_no: str = Field(..., description="Student Registration/Enrollment Number")
    name: str = Field(..., description="Student Name")
    gpa: Optional[str] = Field(None, description="Grade Point Average")
    subjects: List[Subject]

class CertificateData(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    certificate_no: str = Field(..., description="Certificate Number (e.g., top right number)")
    no: str = Field(..., description="Reference Number (e.g., bottom left No. suffix)")
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
    credit_points: str = Field(..., description="Total Credit Points")

class SemesterData(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    semester: str = Field(..., description="Semester name in UPPERCASE words")
    gpa: Optional[str] = Field(None, description="GPA for the semester")
    cgpa: Optional[str] = Field(None, description="CGPA (up to this semester)")
    courses: List[Course]

class YearData(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    year: str = Field(..., description="Year level in UPPERCASE words")
    semesters: List[SemesterData]

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
    years: List[YearData]

class ValidationResponse(BaseModel):
    is_valid: bool = Field(..., description="Whether the document meets quality standards")
    instruction: str = Field(..., description="User-friendly instruction for the pop-up")
    file_type: Optional[str] = None