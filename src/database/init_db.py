# src/database/init_db.py

import sys
import os

current_path = os.path.abspath(__file__)
db_folder = os.path.dirname(current_path)       # src/database
src_folder = os.path.dirname(db_folder)         # src
project_root = os.path.dirname(src_folder)      # project_root

# 프로젝트 루트를 파이썬 라이브러리 경로에 추가
sys.path.append(project_root)

from src.database.connection import db_manager

def init_database():
    print("🚀 데이터베이스 초기화를 시작합니다...")
    
    try:
        # 이 함수가 models.py를 읽어서 테이블이 없으면 생성합니다.
        db_manager.create_all_tables()
        print("✅ 초기화 완료! 모든 테이블이 생성되었습니다.")
        
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        print("config.py의 DB 접속 정보가 정확한지 확인해주세요.")

if __name__ == "__main__":
    init_database()