# src/modules/z_score_updater.py

import numpy as np
from sqlalchemy.orm import Session
from tqdm import tqdm # 진행 현황 파악용
from src.database import crud
from src.utils.logger import setup_experiment_logger

class ZScoreUpdater:
    def __init__(self, session: Session):
        """
        [3단계: 도메인 내 상대적 중요도(Z-Score) 업데이트]
        - 도메인 내부에서 해당 단어가 얼마나 유의미하게 높은 TF-IDF를 가지는지 정규화합니다.
        """
        self.session = session
        self.logger = setup_experiment_logger(experiment_code="Z_SCORE_UPDATER")

    def update_z_scores(self):
        """도메인 내 TF-IDF 점수들의 평균과 표준편차를 사용하여 Z-Score를 산출합니다."""
        domains = crud.get_all_domains(self.session)
        
        # [진행바] 도메인별 Z-Score 계산 시작
        for domain in tqdm(domains, desc="[Step 3] Normalizing Z-Scores", unit="domain"):
            d_id = domain['domain_id']
            # 도메인 데이터 로드
            dtm_rows = list(crud.get_dtm_by_domain(self.session, d_id))
            if not dtm_rows: continue
            
            # NumPy를 활용한 고속 통계 연산
            tfidf_vals = np.array([row['tfidf_score'] for row in dtm_rows])
            mean_v = np.mean(tfidf_vals)
            std_v = np.std(tfidf_vals)
            
            # 표준편차가 0인 경우(모든 점수 동일) 대비
            if std_v == 0: std_v = 1.0
            
            update_payload = []
            for row in dtm_rows:
                # Z = (x - mean) / std
                z = (row['tfidf_score'] - mean_v) / std_v
                update_payload.append({
                    'domain_id': d_id,
                    'term': row['term'],
                    'z_score': float(z)
                })
            
            # Z-Score 최종 반영
            if update_payload:
                crud.bulk_update_dtm_items(self.session, update_payload)