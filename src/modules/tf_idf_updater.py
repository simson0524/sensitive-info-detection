# src/modules/tf_idf_updater.py

import math
from sqlalchemy.orm import Session
from tqdm import tqdm
from src.database import crud
from src.utils.logger import setup_experiment_logger

class TFIDFUpdater:
    def __init__(self, session: Session):
        """
        [2단계: TF-IDF 점수 업데이트 및 통계 누적]
        - IDF를 계산하고 DTM의 TF-IDF 점수를 갱신합니다.
        - 중요: 신규 도메인의 점수를 term 테이블의 누적 합계(sum, sum_square)에 반영합니다.
        """
        self.session = session
        self.logger = setup_experiment_logger(experiment_code="TF_IDF_UPDATER")

    def update_tfidf_scores(self):
        """전체 도메인 개수를 바탕으로 IDF를 산출하고 전역 통계를 누적 업데이트합니다."""
        
        # 1. 전체 도메인 수(N) 확보
        total_n = crud.get_domain_count(self.session)
        if total_n == 0:
            self.logger.warning("도메인이 없습니다. 계산을 중단합니다.")
            return

        # 2. 전역 단어장 정보를 메모리에 로드 (sum_tfidf 등을 업데이트하기 위함)
        self.logger.info("전역 단어 통계 데이터를 로드 중...")
        term_map = {t['term']: t for t in crud.get_all_terms_streaming(self.session)}

        # 3. 이번 실행에서 계산할 대상 도메인 추출 
        # (이미 초기화된 DTM에 tf_score는 있고 tfidf_score가 0인 데이터들 위주)
        domains = crud.get_all_domains(self.session)
        
        # 단어별 이번 턴 누적치를 임시 저장할 딕셔너리 {term: {'sum': 0.0, 'sq_sum': 0.0}}
        incremental_stats = {term: {'sum': 0.0, 'sq_sum': 0.0} for term in term_map.keys()}

        for domain in tqdm(domains, desc="[Step 2] TF-IDF & Stats Accumulation", unit="domain"):
            d_id = domain['domain_id']
            dtm_rows = list(crud.get_dtm_by_domain(self.session, d_id))
            
            # 이번 도메인에서 처리된 결과들을 담을 리스트
            dtm_update_list = []
            
            for row in dtm_rows:
                term = row['term']
                if term not in term_map: continue
                
                # IDF 계산: log10( N / (1 + df) )
                df = term_map[term]['included_domain_counts']
                idf = math.log10(total_n / (1 + df))
                tfidf = row['tf_score'] * idf
                
                dtm_update_list.append({
                    'domain_id': d_id,
                    'term': term,
                    'idf_score': idf,
                    'tfidf_score': tfidf
                })
                
                # [핵심] 신규 도메인의 점수를 임시 누적 (Z-score를 위한 기초 데이터)
                if term in incremental_stats:
                    incremental_stats[term]['sum'] += tfidf
                    incremental_stats[term]['sq_sum'] += (tfidf ** 2)

            # DTM 테이블에 계산된 IDF, TF-IDF 점수 반영
            if dtm_update_list:
                crud.bulk_update_dtm_items(self.session, dtm_update_list)

        # 4. Term 테이블의 전역 통계(sum, sum_square) 누적 업데이트
        self.logger.info("Term 테이블 전역 통계 누적 업데이트 중...")
        term_bulk_updates = []
        for term, stats in incremental_stats.items():
            if stats['sum'] == 0: continue # 이번에 등장하지 않은 단어는 패스
            
            # 기존 합계에 이번 신규 도메인들의 합계를 더함
            new_sum = term_map[term]['sum_tfidf'] + stats['sum']
            new_sq_sum = term_map[term]['sum_square_tfidf'] + stats['sq_sum']
            
            term_bulk_updates.append({
                'term': term,
                'sum_tfidf': new_sum,
                'sum_square_tfidf': new_sq_sum
            })
            
        if term_bulk_updates:
            # 주의: 여기서는 avg와 stddev는 아직 계산하지 않습니다. 
            # 모든 도메인의 합산이 완료된 최종 단계(ZScoreUpdater)에서 한 번에 계산하는 것이 더 정확합니다.
            crud.bulk_update_terms(self.session, term_bulk_updates)
        
        self.session.commit()
        self.logger.info("TF-IDF 업데이트 및 통계 누적 완료.")