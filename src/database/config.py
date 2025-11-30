# src/database/config.py

### ------------------------------------------------------------
### Database 기본 설정
### ------------------------------------------------------------
DATABASE_HOST     = "127.0.0.1"
DATABASE_PORT     = "55432"
DATABASE_DBNAME   = "postgres"
DATABASE_USER     = "student1"
DATABASE_PASSWORD = "onestone"

# SQLAlchemy용 접속 URL 생성
def get_db_url():
    return f"postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_DBNAME}"