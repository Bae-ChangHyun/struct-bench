from pydantic import BaseModel, Field


class WorkExperience(BaseModel):
    company: str = Field(description="회사명")
    title: str = Field(description="직함/직책")
    start_date: str = Field(description="시작일 (YYYY-MM 형식)")
    end_date: str | None = Field(default=None, description="종료일 (YYYY-MM 형식, 현재 재직 중이면 null)")
    description: str = Field(default="", description="업무 설명")


class Education(BaseModel):
    institution: str = Field(description="교육기관명")
    degree: str = Field(description="학위 (예: Bachelor, Master)")
    field_of_study: str = Field(default="", description="전공")
    graduation_date: str | None = Field(default=None, description="졸업일 (YYYY-MM 형식)")


class Resume(BaseModel):
    name: str = Field(description="이름")
    email: str | None = Field(default=None, description="이메일")
    phone: str | None = Field(default=None, description="전화번호")
    summary: str = Field(default="", description="자기소개 요약")
    skills: list[str] = Field(default_factory=list, description="기술 스택 목록")
    work_experience: list[WorkExperience] = Field(default_factory=list, description="경력 목록")
    education: list[Education] = Field(default_factory=list, description="학력 목록")
