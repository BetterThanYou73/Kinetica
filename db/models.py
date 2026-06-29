from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()


class Session(Base):
    __tablename__ = "sessions"
    id         = Column(Integer, primary_key=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at   = Column(DateTime, nullable=True)
    equipment  = Column(String)  # comma-separated


class Set(Base):
    __tablename__ = "sets"
    id         = Column(Integer, primary_key=True)
    session_id = Column(Integer)
    exercise   = Column(String)
    reps       = Column(Integer)
    weight_kg  = Column(Float, nullable=True)
    logged_at  = Column(DateTime, default=datetime.utcnow)


class PersonalRecord(Base):
    __tablename__ = "personal_records"
    id         = Column(Integer, primary_key=True)
    exercise   = Column(String)
    metric     = Column(String)   # "max_reps", "max_weight", "fastest_time"
    value      = Column(Float)
    achieved_at = Column(DateTime, default=datetime.utcnow)


def init_db(path="kinetica.db"):
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
