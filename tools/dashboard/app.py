# tools/dashboard/app.py

import streamlit as st
import sys
import os

# 프로젝트 루트 경로 추가 (src 모듈 사용 위함)
# tools/dashboard/app.py 기준 2단계 상위가 루트
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

st.set_page_config(
    page_title="Experiment Dashboard",
    page_icon="🎨",
    layout="wide"
)

st.title("실험 파이프라인 대시보드")
st.markdown("""
이 대시보드는 **실험 결과 모니터링**과 **데이터 라벨링(Human-in-the-loop)**을 지원합니다.
왼쪽 사이드바에서 메뉴를 선택하세요.

- **📊 Metric Viewer**: 실험별 Loss 그래프, 성능 지표 확인
- **🏷️ Candidate Labeler**: 모델이 헷갈려하는 데이터 직접 수정 (DB 반영)
""")


## 미완성(패키지 파일 미완성) ##