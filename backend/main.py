import os
from typing import Annotated

from fastapi import FastAPI, Depends, HTTPException, status, Form
from pydantic import ValidationError
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import FileResponse
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session
from jose import jwt, JWTError

from models import Base, User, Submission
from auth import verify_password, get_password_hash, create_access_token, oauth2_scheme, SECRET_KEY, ALGORITHM
from schemas import SubmissionCreate, SubmissionResponse, UserCreate

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://bench_user:super_secret_password@db:5432/bench_db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Bench API")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

DbSession = Annotated[Session, Depends(get_db)]

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
    username: Annotated[str, Form(...)], 
    password: Annotated[str, Form(...)], 
    db: DbSession
):
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


@app.get("/api/downloads/archive")
async def download_archive():
    file_path = "dummy_archive.zip"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Archive not found")
    return FileResponse(file_path, media_type="application/zip", filename="env_tools.zip")


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

    return {"status": "success", "id": new_submission.id}


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