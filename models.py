from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional

class Subject(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    code: str = Field(..., description="Course code")
    title: str = Field(..., description="Course title")
    credit_points: str = Field(..., description="Total Credit Points")
    # grade: str = Field(..., description="Grade Points (e.g., '8.8')")

class MarkSheetData(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    registration_no: str = Field(..., description="Student Registration/Enrollment Number")
    name: str = Field(..., description="Student Name")
    gpa: Optional[str] = Field(None, description="Grade Point Average")
    subjects: List[Subject]

class ValidationResponse(BaseModel):
    is_valid: bool = Field(..., description="Whether the document meets quality standards")
    instruction: str = Field(..., description="User-friendly instruction for the pop-up")
    file_type: Optional[str] = None