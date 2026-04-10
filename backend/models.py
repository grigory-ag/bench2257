from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class Submission(Base):
    __tablename__ = "submissions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    language = Column(String, index=True)      
    task_name = Column(String, index=True)     
    attempt_number = Column(Integer)           
    
    time_ms = Column(Float)
    cpu_max_ram_mb = Column(Float)             
    gpu_max_ram_mb = Column(Float)             
    cpu_model = Column(String)                 
    gpu_model = Column(String)                 
    
    archive_path = Column(String)              