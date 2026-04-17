from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional

class Subject(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    code: str = Field(..., description="Course code")
    title: str = Field(..., description="Course title")
    credits: str = Field(..., description="Credit hours")
    grade: str = Field(..., description="Grade awarded")
    credit_points: Optional[str] = None
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

class ValidationResponse(BaseModel):
    is_valid: bool = Field(..., description="Whether the document meets quality standards")
    instruction: str = Field(..., description="User-friendly instruction for the pop-up")
    file_type: Optional[str] = None