"""description이 없는 Career 스키마"""
from pydantic import BaseModel
from typing import List, Optional, Literal


class CareerNoDesc(BaseModel):
    company_name: Optional[str] = ""
    is_company_private: Optional[bool] = ""
    start_date: Optional[str] = ""
    end_date: Optional[str] = ""
    is_currently_employed: Optional[bool] = ""
    department: Optional[str] = ""
    position: Optional[str] = ""
    work_details: Optional[str] = ""
    annual_salary: Optional[str] = ""
    reason_for_leaving: Optional[str] = ""
    employment_type: Optional[str] = ""
    work_location: Optional[str] = ""


class ActivityExperienceNoDesc(BaseModel):
    activity_type: Optional[str] = ""
    activity_name: Optional[str] = ""
    organization: Optional[str] = ""
    start_date: Optional[str] = ""
    end_date: Optional[str] = ""
    details: Optional[str] = ""


class OverseasExperienceNoDesc(BaseModel):
    experience_type: Optional[Literal["어학연수", "교환학생", "워킹홀리데이", "유학"]] = ""
    country: Optional[str] = ""
    start_date: Optional[str] = ""
    end_date: Optional[str] = ""
    details: Optional[str] = ""


class LanguageSkillNoDesc(BaseModel):
    assessment_type: Optional[Literal["회화능력", "공인시험"]] = ""
    language: Optional[str] = ""
    proficiency_level: Optional[str] = ""
    test_name: Optional[str] = ""
    test_language: Optional[str] = ""
    test_score: Optional[str] = ""
    test_date: Optional[str] = ""


class CertificateNoDesc(BaseModel):
    certificate_name: Optional[str] = ""
    issuing_authority: Optional[str] = ""
    acquisition_date: Optional[str] = ""


class AwardExperienceNoDesc(BaseModel):
    award_name: Optional[str] = ""
    organizer: Optional[str] = ""
    award_date: Optional[str] = ""
    details: Optional[str] = ""


class EmploymentAndMilitaryInfoNoDesc(BaseModel):
    is_veteran_target: Optional[bool] = ""
    veteran_reason: Optional[str] = ""
    is_employment_protection_target: Optional[bool] = ""
    is_disabled: Optional[bool] = ""
    disability_grade: Optional[str] = ""
    military_status: Optional[Literal["군필", "미필", "면제", "해당없음"]] = ""
    service_start_date: Optional[str] = ""
    service_end_date: Optional[str] = ""
    military_branch: Optional[str] = ""
    rank: Optional[str] = ""


class OnlineProfileNoDesc(BaseModel):
    sns_links: List[str] = []


class MainInfoNoDesc(BaseModel):
    careers: List[CareerNoDesc] = []
    activity_experiences: List[ActivityExperienceNoDesc] = []
    overseas_experiences: List[OverseasExperienceNoDesc] = []
    language_skills: List[LanguageSkillNoDesc] = []
    certificates: List[CertificateNoDesc] = []
    award_experiences: List[AwardExperienceNoDesc] = []
    employment_military_info: Optional[EmploymentAndMilitaryInfoNoDesc] = ""
    sns: Optional[OnlineProfileNoDesc] = ""
