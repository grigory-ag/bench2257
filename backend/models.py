from sqlalchemy import Column, Integer, String, Float, ForeignKey, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_admin = Column(Boolean, default=False)

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
    source_code = Column(Text, nullable=True)
    status = Column(String, default="pending")

# Паспорт задачи (будущее расширение БД): для каждой task_name хранить
# краткое описание, подробное ТЗ, технические требования и автора эталона (например, "Core Team").
# Пример ORM: TaskSpec(task_key PK, summary, spec_md, technical_md, reference_author).
# Пока эти поля задаются на фронтенде (theory.html, объект TASK_PASSPORT).