from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    username: str
    password: str = Field(..., min_length=6)


class SubmissionCreate(BaseModel):
    language: str
    task_name: str
    time_ms: float
    cpu_max_ram_mb: float
    gpu_max_ram_mb: float
    cpu_model: str
    gpu_model: str
    source_code: str | None = Field(default=None, max_length=50000)


class SubmissionResponse(BaseModel):
    status: str
    id: int


class UserRead(BaseModel):
    id: int
    username: str
    is_admin: bool


class SubmissionStatusUpdate(BaseModel):
    status: str
