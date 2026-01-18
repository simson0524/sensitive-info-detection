# src/modules/z_score_updater.py

import numpy as np
import math
from sqlalchemy.orm import Session
from tqdm import tqdm
from src.database import crud
from src.utils.logger import setup_experiment_logger

class ZScoreUpdater:
    def __init__(self, session: Session):
        """
        [Phase 3: 전역 통계 기반 Z-Score 산출 및 통계 확정]
        - 이 모듈은 Term 테이블에 누적된 sum, sum_square 데이터를 바탕으로 최종 통계치를 도출합니다.
        - DTM 테이블에는 '자기 자신을 제외(Leave-one-out)'한 상대적 Z-score를 기록하고,
        - Term 테이블에는 '시스템 전체'를 기준으로 한 절대적 평균/표준편차를 기록합니다.
        """
        self.session = session
        self.logger = setup_experiment_logger(experiment_code="Z_SCORE_UPDATER")

    def update_z_scores(self):
        """
        전역 누적 통계를 바탕으로 각 도메인의 단어별 Z-score를 산출합니다.
        'Leave-one-out' 방식을 적용하여 특정 도메인의 값이 전체 평균을 왜곡하는 현상을 방지합니다.
        """
        # 1. 전역 파라미터 확보 (N = 총 도메인 개수)
        total_n = crud.get_domain_count(self.session)
        if total_n <= 1:
            self.logger.warning("비교 대상이 부족하여 Z-score를 산출할 수 없습니다. (N <= 1)")
            return

        # 2. 전역 단어 통계(누적 합계/제곱합) 로드
        # 이전 단계에서 신규 데이터까지 모두 포함되어 업데이트된 마스터 통계치를 가져옵니다.
        self.logger.info("Term 테이블로부터 전역 누적 통계 데이터를 로드 중...")
        term_map = {t['term']: t for t in crud.get_all_terms_streaming(self.session)}

        # 3. 모든 도메인을 순회하며 상대적 Z-score 계산
        domains = crud.get_all_domains(self.session)
        
        for domain in tqdm(domains, desc="[Step 3] Calculating Leave-one-out Z-Score"):
            d_id = domain['domain_id']
            dtm_rows = list(crud.get_dtm_by_domain(self.session, d_id))
            if not dtm_rows: continue
            
            update_payload = []
            for row in dtm_rows:
                term = row['term']
                if term not in term_map: continue
                
                # [전역 누적 데이터 확보]
                g_sum = term_map[term]['sum_tfidf']         # 모든 도메인의 TF-IDF 총합
                g_sq_sum = term_map[term]['sum_square_tfidf'] # 모든 도메인의 TF-IDF 제곱합
                curr_val = row['tfidf_score']                # 현재 도메인의 TF-IDF 값
                
                # [교수님 요구사항: 자기 자신 제외(Leave-one-out) 로직]
                # 현재 단어가 속한 도메인의 값을 통계에서 제외하여 '배경지식'과의 순수 비교를 수행합니다.
                # 이를 통해 특정 단어가 현재 도메인에서만 유독 튀는 현상을 극대화(민감정보 탐지 유리)합니다.
                
                n_prime = total_n - 1  # 나를 제외한 표본 개수
                
                # 1) 나를 제외한 평균 역산 (mu')
                mu_prime = (g_sum - curr_val) / n_prime
                
                # 2) 나를 제외한 분산 역산 (var') 
                # 공식: E[X^2] - (E[X])^2를 활용 (제곱의 평균 - 평균의 제곱)
                # 부동 소수점 오차로 인한 음수 발생 방지를 위해 max(0, ...) 처리
                var_prime = ((g_sq_sum - (curr_val**2)) / n_prime) - (mu_prime**2)
                std_prime = math.sqrt(max(0, var_prime))
                
                # 3) Z-score 산출: (현재값 - 나를 제외한 평균) / 나를 제외한 표준편차
                if std_prime == 0:
                    # 모든 도메인에서 동일한 TF-IDF 값을 갖거나 나 외에 데이터가 없는 경우
                    z = 0.0
                else:
                    z = (curr_val - mu_prime) / std_prime
                
                update_payload.append({
                    'domain_id': d_id,
                    'term': term,
                    'z_score': float(z)
                })
            
            # 현재 도메인의 Z-score 정보를 DTM 테이블에 일괄 업데이트
            if update_payload:
                crud.bulk_update_dtm_items(self.session, update_payload)
        
        # 4. Term 테이블 마스터 정보 업데이트
        # 개별 도메인 비교용이 아닌, 시스템 전체 관점에서의 단어별 평균/표준편차를 확정합니다.
        self.logger.info("Term 테이블의 전역 평균/표준편차 마스터 정보를 확정 중...")
        self._update_term_global_stats(total_n, term_map)
        
        self.session.commit()
        self.logger.info("Z-score 파이프라인 최종 완료.")

    def _update_term_global_stats(self, n, term_map):
        """
        시스템 전체(N)를 기준으로 한 단어별 절대적 통계치를 계산하여 Term 테이블에 저장합니다.
        이 데이터는 향후 분석이나 대시보드 시각화의 기본 지표로 활용됩니다.
        """
        term_updates = []
        for term, data in term_map.items():
            # 전체 합계를 N으로 나누어 단순 전역 평균 계산
            avg = data['sum_tfidf'] / n
            
            # 전체 제곱합을 활용하여 전역 분산 및 표준편차 계산
            var = (data['sum_square_tfidf'] / n) - (avg**2)
            std = math.sqrt(max(0, var))
            
            term_updates.append({
                'term': term,
                'avg_tfidf': avg,
                'stddev_tfidf': std
            })
            
        if term_updates:
            # Term 테이블의 avg_tfidf, stddev_tfidf 컬럼을 일괄 업데이트하는 CRUD 함수
            crud.bulk_update_terms(self.session, term_updates)