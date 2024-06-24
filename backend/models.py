from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

Base = declarative_base()

class UserSetting(Base):
    __tablename__ = 'user_settings'
    id = Column(Integer, primary_key=True, index=True)
    confidence_threshold = Column(Float, default=0.5)
    model_name = Column(String, default="SSD")

class DetectionHistory(Base):
    __tablename__ = 'detection_history'
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime)
    object_name = Column(String)
    confidence = Column(Float)
    image_path = Column(String)

class ClassStatistic(Base):
    __tablename__ = 'class_statistics'
    id = Column(Integer, primary_key=True, index=True)
    class_name = Column(String)
    count = Column(Integer)

engine = create_engine('sqlite:///./object_detection.db')
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
