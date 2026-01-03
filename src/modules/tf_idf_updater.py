# src/modules/tf_idf_updater.py

import math
from sqlalchemy.orm import Session
from tqdm import tqdm # 진행 현황 파악용
from src.database import crud
from src.utils.logger import setup_experiment_logger

class TFIDFUpdater:
    def __init__(self, session: Session):
        """
        [2단계: TF-IDF 점수 업데이트]
        - 1단계에서 적재된 통계치를 바탕으로 IDF와 TF-IDF 점수를 산출합니다.
        """
        self.session = session
        self.logger = setup_experiment_logger(experiment_code="TF_IDF_UPDATER")

    def update_tfidf_scores(self):
        """전체 도메인을 순회하며 각 단어의 전역 중요도(IDF)를 반영합니다."""
        total_n = crud.get_domain_count(self.session)
        if total_n == 0:
            self.logger.warning("No domains found. Skipping TF-IDF calculation.")
            return

        domains = crud.get_all_domains(self.session)
        # [진행바] 도메인별 TF-IDF 계산 시작
        for domain in tqdm(domains, desc="[Step 2] Updating TF-IDF Scores", unit="domain"):
            d_id = domain['domain_id']
            update_list = []
            
            # 도메인에 속한 모든 단어 행 조회
            dtm_rows = list(crud.get_dtm_by_domain(self.session, d_id))
            for row in dtm_rows:
                # Term 통계(df) 조회
                term_info = crud.get_term_stats(self.session, row['term'])
                df = term_info['included_domain_counts']
                
                # IDF 공식: log10( N / (1 + df) )
                idf = math.log10(total_n / (1 + df))
                tfidf = row['tf_score'] * idf
                
                update_list.append({
                    'domain_id': d_id,
                    'term': row['term'],
                    'idf_score': idf,
                    'tfidf_score': tfidf
                })
            
            # 계산된 점수 일괄 업데이트
            if update_list:
                crud.bulk_update_dtm_items(self.session, update_list)