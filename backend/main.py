import os
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Depends, HTTPException, status, Form, Request
from pydantic import ValidationError
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse
from sqlalchemy import create_engine, func, or_
from sqlalchemy.orm import sessionmaker, Session
from jose import jwt, JWTError

from models import Base, User, Submission
from auth import verify_password, get_password_hash, create_access_token, oauth2_scheme, SECRET_KEY, ALGORITHM
from schemas import SubmissionCreate, SubmissionResponse, UserCreate, UserRead, SubmissionStatusUpdate

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://bench_user:super_secret_password@db:5432/bench_db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Bench API")
registration_attempts: dict[str, list[float]] = {}

# Имена задач, совпадающие с клиентом бенчмарка и frontend/js/task-translations.js
KNOWN_TASK_NAMES = frozenset(
    {
        "Sum of Matrices",
        "Multiply Matrices",
        "Invert Matrices",
        "Random Walk",
        "Fractals",
        "Zombie Apocalypse",
        "Pandemic Spread",
    }
)


def _scripts_root() -> Path:
    """Путь к benchmark_client/scripts: рядом с main.py (Docker) или на уровень выше (локально из backend/)."""
    here = Path(__file__).resolve().parent
    docker_style = here / "benchmark_client" / "scripts"
    if docker_style.is_dir():
        return docker_style
    repo_style = here.parent / "benchmark_client" / "scripts"
    if repo_style.is_dir():
        return repo_style
    return docker_style


def _benchmark_client_root() -> Path:
    here = Path(__file__).resolve().parent
    docker_style = here / "benchmark_client"
    if docker_style.is_dir():
        return docker_style
    repo_style = here.parent / "benchmark_client"
    if repo_style.is_dir():
        return repo_style
    return docker_style


def _is_safe_task_name(name: str) -> bool:
    if not name or name.strip() != name:
        return False
    if ".." in name or "/" in name or "\\" in name:
        return False
    return True


def _read_script_file(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


DbSession = Annotated[Session, Depends(get_db)]


def _get_approved_submission_code(db: Session, task_name: str, language: str) -> dict[str, str] | None:
    row = (
        db.query(Submission.source_code, User.username)
        .join(User, Submission.user_id == User.id)
        .filter(
            Submission.task_name == task_name,
            Submission.language == language,
            Submission.status == "approved",
            Submission.source_code.isnot(None),
        )
        .order_by(Submission.time_ms.asc())
        .first()
    )
    if row is None:
        return None
    return {"code": row.source_code, "author": row.username}


@app.get("/api/tasks/{task_name}/code")
async def get_task_reference_code(task_name: str, db: DbSession):
    """
    Публичная выдача эталонных скриптов из benchmark_client/scripts/
    (python/cpp/cuda/go, имя файла = имя задачи + расширение).
    """
    if not _is_safe_task_name(task_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid task name",
        )
    if task_name not in KNOWN_TASK_NAMES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Unknown task",
        )

    root = _scripts_root()
    # В репозитории каталоги: python, cpp, cuda, go (нижний регистр)
    mapping = (
        ("python", root / "python" / f"{task_name}.py"),
        ("cpp", root / "cpp" / f"{task_name}.cpp"),
        ("cuda", root / "cuda" / f"{task_name}.cu"),
        ("go", root / "go" / f"{task_name}.go"),
        ("matlab", root / "matlab" / f"{task_name}.m"),
    )

    result = {}
    for language, path in mapping:
        submission_code = _get_approved_submission_code(db, task_name, language)
        if submission_code is not None:
            result[language] = submission_code
            continue

        file_code = _read_script_file(path)
        result[language] = (
            {"code": file_code, "author": "Core Team"}
            if file_code is not None
            else None
        )

    return result

def get_current_user(token: Annotated[str, Depends(oauth2_scheme)], db: DbSession):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

CurrentUser = Annotated[User, Depends(get_current_user)]

@app.post("/auth/register")
async def register(
    request: Request,
    username: Annotated[str, Form(...)], 
    password: Annotated[str, Form(...)], 
    db: DbSession
):
    now = time.time()
    ip_address = request.client.host if request.client else "unknown"
    recent_attempts = [
        ts for ts in registration_attempts.get(ip_address, [])
        if now - ts < 3600
    ]
    if len(recent_attempts) >= 5:
        registration_attempts[ip_address] = recent_attempts
        raise HTTPException(
            status_code=429,
            detail="Слишком много попыток регистрации. Попробуйте позже.",
        )
    recent_attempts.append(now)
    registration_attempts[ip_address] = recent_attempts

    try:
        UserCreate(username=username, password=password)
    except ValidationError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters",
        )

    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Username already registered"
        )
    
    try:
        hashed_pw = get_password_hash(password)
        new_user = User(username=username, hashed_password=hashed_pw)
        
        db.add(new_user)
        db.commit()   
        db.refresh(new_user) 
        
        return {"status": "ok", "message": "User created successfully"}
    
    except Exception as e:
        db.rollback() 
        print(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Internal server error during registration"
        )


@app.post("/auth/token")
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()], 
    db: DbSession
):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Incorrect username or password"
        )
        
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/auth/me", response_model=UserRead)
async def auth_me(current_user: CurrentUser):
    return UserRead(
        id=current_user.id,
        username=current_user.username,
        is_admin=bool(current_user.is_admin),
    )


@app.get("/api/analytics/raw")
async def get_analytics_raw(db: DbSession):
    """Публичные сырые данные: официальные (без кода) или одобренные кастомные."""
    rows = (
        db.query(
            Submission.id,
            Submission.task_name,
            Submission.language,
            Submission.time_ms,
            Submission.cpu_model,
            Submission.gpu_model,
            Submission.cpu_max_ram_mb,
            User.username,
        )
        .outerjoin(User, Submission.user_id == User.id)
        .filter(
            or_(
                Submission.source_code.is_(None),
                Submission.status == "approved",
            )
        )
        .order_by(Submission.id.desc())
        .all()
    )
    return [
        {
            "id": r.id,
            "task_name": r.task_name,
            "language": (r.language or "").lower(),
            "time_ms": float(r.time_ms) if r.time_ms is not None else None,
            "cpu_model": r.cpu_model,
            "gpu_model": r.gpu_model,
            "cpu_max_ram_mb": float(r.cpu_max_ram_mb) if r.cpu_max_ram_mb is not None else None,
            "username": r.username,
        }
        for r in rows
    ]


@app.get("/api/analytics/global")
async def get_global_analytics(db: DbSession):
    """Public aggregate: average time_ms per task and language (no auth)."""
    rows = (
        db.query(
            Submission.task_name,
            Submission.language,
            func.avg(Submission.time_ms).label("avg_time_ms"),
        )
        .group_by(Submission.task_name, Submission.language)
        .all()
    )
    return [
        {
            "task_name": r.task_name,
            "language": (r.language or "").lower(),
            "avg_time_ms": float(r.avg_time_ms) if r.avg_time_ms is not None else 0.0,
        }
        for r in rows
    ]


@app.get("/api/leaderboard")
async def get_global_leaderboard(db: DbSession):
    rows = (
        db.query(
            User.username,
            Submission.task_name,
            Submission.language,
            Submission.time_ms,
            Submission.cpu_model,
            Submission.source_code,
        )
        .join(User, Submission.user_id == User.id)
        .filter(
            Submission.status == "approved",
            Submission.source_code.isnot(None),
        )
        .order_by(Submission.time_ms.asc())
        .limit(50)
        .all()
    )

    return [
        {
            "username": row.username,
            "task_name": row.task_name,
            "language": row.language,
            "time_ms": row.time_ms,
            "cpu_model": row.cpu_model,
            "source_code": row.source_code,
        }
        for row in rows
    ]


@app.get("/api/downloads/archive")
async def download_archive():
    client_root = _benchmark_client_root()
    if not client_root.is_dir():
        raise HTTPException(status_code=404, detail="Benchmark client directory not found")

    excluded_dirs = {"build", "dist", "venv", "__pycache__"}
    excluded_files = {".env"}
    excluded_extensions = {".dat", ".csv", ".bin", ".spec"}
    buffer = BytesIO()

    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in client_root.rglob("*"):
            relative_path = path.relative_to(client_root)
            parts = set(relative_path.parts)
            if parts.intersection(excluded_dirs):
                continue
            if path.name in excluded_files:
                continue
            if path.suffix.lower() in excluded_extensions:
                continue
            if path.is_file():
                archive.write(path, relative_path.as_posix())

    buffer.seek(0)
    headers = {"Content-Disposition": 'attachment; filename="BenchmarkSystem.zip"'}
    return StreamingResponse(buffer, media_type="application/zip", headers=headers)


@app.post("/api/submissions", response_model=SubmissionResponse)
async def create_submission(
    payload: SubmissionCreate,
    current_user: CurrentUser,
    db: DbSession,
):
    max_attempt = db.query(func.max(Submission.attempt_number)).filter(
        Submission.user_id == current_user.id,
        Submission.language == payload.language,
        Submission.task_name == payload.task_name,
    ).scalar()

    attempt_number = (max_attempt or 0) + 1

    new_submission = Submission(
        user_id=current_user.id,
        language=payload.language,
        task_name=payload.task_name,
        attempt_number=attempt_number,
        time_ms=payload.time_ms,
        cpu_max_ram_mb=payload.cpu_max_ram_mb,
        gpu_max_ram_mb=payload.gpu_max_ram_mb,
        cpu_model=payload.cpu_model,
        gpu_model=payload.gpu_model,
        archive_path=None,
        source_code=payload.source_code,
        status="pending" if payload.source_code else "approved",
    )

    try:
        db.add(new_submission)
        db.commit()
        db.refresh(new_submission)
    except Exception as e:
        db.rollback()
        print(f"Submission error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save submission",
        )

    return {"status": new_submission.status, "id": new_submission.id}


@app.get("/api/submissions/{language}/{task_name}")
async def get_submissions(
    language: str, 
    task_name: str, 
    current_user: CurrentUser, 
    db: DbSession
):
    subs = db.query(Submission).filter(
        Submission.user_id == current_user.id,
        Submission.language == language,
        Submission.task_name == task_name
    ).order_by(Submission.attempt_number.asc()).all()
    return subs


@app.get("/api/admin/submissions")
async def get_admin_submissions(
    current_user: CurrentUser,
    db: DbSession,
):
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    rows = (
        db.query(
            Submission.id,
            User.username,
            Submission.task_name,
            Submission.language,
            Submission.time_ms,
            Submission.source_code,
            Submission.status,
        )
        .join(User, Submission.user_id == User.id)
        .filter(Submission.source_code.isnot(None))
        .order_by(Submission.id.desc())
        .all()
    )
    return [
        {
            "id": row.id,
            "username": row.username,
            "task_name": row.task_name,
            "language": row.language,
            "time_ms": row.time_ms,
            "source_code": row.source_code,
            "status": row.status,
        }
        for row in rows
    ]


@app.patch("/api/admin/submissions/{sub_id}/status")
async def update_submission_status(
    sub_id: int,
    payload: SubmissionStatusUpdate,
    current_user: CurrentUser,
    db: DbSession,
):
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    if payload.status not in ("approved", "rejected"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Status must be approved or rejected",
        )

    submission = db.query(Submission).filter(Submission.id == sub_id).first()
    if submission is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Submission not found",
        )

    submission.status = payload.status
    db.commit()
    db.refresh(submission)
    return {"id": submission.id, "status": submission.status}
