"""description이 없는 Resume 스키마"""
from pydantic import BaseModel


class WorkExperienceNoDesc(BaseModel):
    company: str
    title: str
    start_date: str
    end_date: str | None = None
    description: str = ""


class EducationNoDesc(BaseModel):
    institution: str
    degree: str
    field_of_study: str = ""
    graduation_date: str | None = None


class ResumeNoDesc(BaseModel):
    name: str
    email: str | None = None
    phone: str | None = None
    summary: str = ""
    skills: list[str] = []
    work_experience: list[WorkExperienceNoDesc] = []
    education: list[EducationNoDesc] = []
