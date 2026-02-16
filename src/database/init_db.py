# src/database/init_db.py

import sys
import os

current_path = os.path.abspath(__file__)
db_folder = os.path.dirname(current_path)       # src/database
src_folder = os.path.dirname(db_folder)         # src
project_root = os.path.dirname(src_folder)      # project_root

# 프로젝트 루트를 파이썬 라이브러리 경로에 추가
sys.path.append(project_root)

# from src.database.connection import db_manager

# def init_database():
#     print("🚀 데이터베이스 초기화를 시작합니다...")
    
#     try:
#         # 이 함수가 models.py를 읽어서 테이블이 없으면 생성합니다.
#         db_manager.create_all_tables()
#         print("✅ 초기화 완료! 모든 테이블이 생성되었습니다.")
        
#     except Exception as e:
#         print(f"❌ 초기화 실패: {e}")
#         print("config.py의 DB 접속 정보가 정확한지 확인해주세요.")

# if __name__ == "__main__":
#     init_database()

import sys
import os
import subprocess
import time
import socket

current_path = os.path.abspath(__file__)
db_folder = os.path.dirname(current_path)       # src/database
src_folder = os.path.dirname(db_folder)         # src
project_root = os.path.dirname(src_folder)      # project_root

sys.path.append(project_root)

from src.database.connection import db_manager
from src.database.config import DATABASE_HOST, DATABASE_PORT

# [경로 설정] 프로젝트 루트 아래 data/db_storage 폴더를 사용합니다.
DB_DATA_PATH = os.path.join(project_root, "data", "db_storage")
LOG_DIR = os.path.join(project_root, "logs")
POSTGRES_LOG = os.path.join(LOG_DIR, "postgres.log")

def is_db_listening(host, port):
    """포트가 열려있는지 확인"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            return s.connect_ex((host, int(port))) == 0
    except Exception:
        return False

def ensure_db_running():
    """DB 데이터가 없으면 초기화하고, 서버가 꺼져있으면 가동"""
    # 1. 포트가 이미 열려있다면 통과
    if is_db_listening(DATABASE_HOST, DATABASE_PORT):
        print(f"✅ DB 서버가 이미 {DATABASE_PORT} 포트에서 작동 중입니다.")
        return

    # 2. 데이터 폴더가 없으면 초기화(initdb) 수행
    PG_BIN = "/usr/lib/postgresql/15/bin"

    if not os.path.exists(os.path.join(DB_DATA_PATH, "PG_VERSION")):
        print(f"✨ DB 데이터 폴더가 없습니다. 초기화를 시작합니다: {DB_DATA_PATH}")
        os.makedirs(DB_DATA_PATH, exist_ok=True)
        try:
            # initdb 실행
            subprocess.run(f"{PG_BIN}/initdb -D {DB_DATA_PATH}", shell=True, check=True)
            print("✅ DB 저장소 초기화(initdb) 완료.")
        except Exception as e:
            print(f"❌ DB 초기화 실패: {e}")
            return

    # 3. 서버 가동
    print(f"⚠️ DB 서버 가동을 시도합니다... (Port: {DATABASE_PORT})")
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 백그라운드 실행 (stdout/stderr는 logs/postgres.log로)
    cmd = f"{PG_BIN}/postgres -D {DB_DATA_PATH} -p {DATABASE_PORT} -k {DB_DATA_PATH} > {POSTGRES_LOG} 2>&1 &"
    subprocess.Popen(cmd, shell=True)
    
    # 서버 대기
    print("⏳ 서버 응답 대기 중", end="", flush=True)
    for _ in range(15):
        if is_db_listening(DATABASE_HOST, DATABASE_PORT):
            print("\n✅ DB 서버 가동 성공!")
            return
        time.sleep(1)
        print(".", end="", flush=True)
    
    print(f"\n❌ 서버 가동 실패. '{POSTGRES_LOG}'를 확인하세요.")

def init_database():
    print("🚀 프로젝트 데이터베이스 초기화 시퀀스 시작...")
    
    ensure_db_running()
    
    try:
        # DB 서버가 뜬 직후에는 연결 준비 시간이 필요할 수 있음
        time.sleep(1)
        db_manager.create_all_tables()
        print("✅ 모든 테이블 스키마가 성공적으로 생성되었습니다.")
    except Exception as e:
        print(f"❌ 테이블 생성 실패: {e}")

if __name__ == "__main__":
    init_database()