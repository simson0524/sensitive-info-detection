# src/database/connection.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager
from src.database.config import get_db_url

# 1. 엔진 생성
engine = create_engine(
    get_db_url(),
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600
)

# 2. 세션 공장 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 3. 모델 베이스 클래스 (모든 모델은 이걸 상속받음)
Base = declarative_base()

class DBManager:
    def __init__(self):
        self._engine = engine
        self._session_factory = SessionLocal

    @contextmanager
    def get_db(self):
        """
        데이터베이스 세션을 관리하는 Context Manager
        사용법: with db_manager.get_db() as session: ...
        """
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def create_all_tables(self):
        """정의된 모든 모델(Base 상속 클래스)을 테이블로 생성"""
        # models가 import 되어 있어야 Base가 인식함
        import src.database.models 
        Base.metadata.create_all(bind=self._engine)
        print("✅ 모든 테이블 스키마가 생성/확인되었습니다.")

# 전역 객체
db_manager = DBManager()