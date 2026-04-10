import os
from typing import Annotated

from fastapi import FastAPI, Depends, HTTPException, status, Form
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import FileResponse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from jose import jwt, JWTError

from models import Base, User, Submission
from auth import verify_password, get_password_hash, create_access_token, oauth2_scheme, SECRET_KEY, ALGORITHM

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


@app.get("/api/downloads/archive")
async def download_archive():
    file_path = "dummy_archive.zip"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Archive not found")
    return FileResponse(file_path, media_type="application/zip", filename="env_tools.zip")


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