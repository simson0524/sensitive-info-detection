# src/modules/z_score_calculator.py

import os
import json
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from src.utils.logger import logger

class ZScoreCalculator:
    """
    [Z-Score Calculator Module]
    
    특정 디렉토리(data_root) 내의 모든 JSON 문서를 로드하여,
    각 단어의 TF-IDF 값을 기반으로 Z-Score(표준점수)를 계산합니다.
    
    기능:
    1. Global Z-Score: 전체 도메인의 모든 문서를 모집단으로 하여 계산 (전역적 중요도)
    2. Local Z-Score: 특정 도메인 내의 문서들만 모집단으로 하여 계산 (지역적 중요도)
    3. 결과 저장: 각 도메인 디렉토리에 'z_score.json' 형태로 저장
    """

    def __init__(self, data_root_dir: str = 'data/train_data'):
        """
        Args:
            data_root_dir (str): 데이터가 위치한 최상위 경로 (기본값: 'data/train_data')
        """
        self.data_root = data_root_dir
        # self.documents: 로드된 모든 문서 정보를 담는 리스트
        # 구조: [{'doc_id': str, 'domain_dir': str, 'full_text': str, ...}, ...]
        self.documents = [] 

    def load_all_data(self):
        """
        [Data Loader]
        data_root 경로 하위에 있는 모든 도메인 폴더를 순회하며 .json 파일을 메모리에 적재합니다.
        
        - 제외 대상: 숨김 폴더(.), 이미 생성된 결과 파일(z_score.json)
        - 전처리: JSON 내부의 'sentence' 필드들을 합쳐 하나의 문자열(full_text)로 변환
        """
        # 경로 안정성을 위해 절대 경로로 변환
        abs_root = os.path.abspath(self.data_root)
        logger.info(f"[ZScore] Scanning data root: {abs_root}")
        
        if not os.path.exists(abs_root):
            logger.error(f"[ZScore] Data root not found: {abs_root}")
            return

        self.documents = [] # 리스트 초기화

        # 1. 도메인 디렉토리 순회 (1-depth)
        for domain_dir in os.listdir(abs_root):
            domain_path = os.path.join(abs_root, domain_dir)
            
            # 디렉토리가 아니거나 숨김 파일(.git 등)은 패스
            if not os.path.isdir(domain_path) or domain_dir.startswith('.'):
                continue
            
            # 2. 각 도메인 폴더 내부의 파일 순회
            loaded_count = 0
            for file_name in os.listdir(domain_path):
                # .json 파일만 처리하되, 결과 파일(z_score.json)은 다시 읽지 않음
                if not file_name.endswith('.json') or file_name == 'z_score.json':
                    continue
                
                file_path = os.path.join(domain_path, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # JSON 구조: [{"sentence": "...", ...}, ...] 가정
                        # TF-IDF 계산을 위해 모든 문장을 공백으로 이어 붙임
                        full_text = " ".join([item.get('sentence', '') for item in data])
                        
                        self.documents.append({
                            'doc_id': file_name.replace('.json', ''),
                            'domain_dir': domain_dir, 
                            'full_text': full_text
                        })
                        loaded_count += 1
                except Exception as e:
                    logger.warning(f"[ZScore] Failed to load {file_name}: {e}")
            
            # (선택사항) 도메인별 로딩 현황 로그 출력
            # logger.debug(f"  - Loaded {loaded_count} docs from {domain_dir}")

        logger.info(f"[ZScore] Total loaded documents: {len(self.documents)}")

    def _compute_stats(self, docs_subset: List[Dict]) -> List[Dict[str, float]]:
        """
        [Core Logic]
        주어진 문서 리스트(Subset)에 대해 TF-IDF 기반 Z-Score를 계산합니다.
        
        Args:
            docs_subset (List[Dict]): 계산 대상 문서 리스트
            
        Returns:
            List[Dict[str, float]]: 각 문서별 {단어: z_score} 딕셔너리 리스트
        """
        if not docs_subset:
            return []

        # 1. 말뭉치(Corpus) 생성
        corpus = [d['full_text'] for d in docs_subset]
        
        # 2. TF-IDF 행렬 생성 (Scikit-learn 활용)
        # token_pattern: 기본적으로 2글자 이상만 잡지만, 한글 단어 등을 고려하여 
        # 모든 단어(\w+)를 잡도록 정규식 수정
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        tfidf_matrix = vectorizer.fit_transform(corpus) # 결과는 Sparse Matrix (희소 행렬)
        feature_names = vectorizer.get_feature_names_out()
        
        # 3. 통계 계산을 위해 Dense Matrix(일반 배열)로 변환
        # (문서 수가 매우 많을 경우 메모리 이슈 주의, 필요 시 배치 처리 고려)
        dense_tfidf = tfidf_matrix.toarray()
        
        # 4. 평균(Mean)과 표준편차(Std) 계산 (Column-wise: 단어별로 계산)
        means = np.mean(dense_tfidf, axis=0)
        stds = np.std(dense_tfidf, axis=0)
        
        # [Zero Division Handling]
        # 표준편차가 0인 경우(모든 문서에서 값이 동일함) 나누기 에러가 발생하므로 1.0으로 대체
        # (Z-score 분모가 1이 되므로 값은 0이 됨)
        stds[stds == 0] = 1.0 
        
        # 5. Z-Score 행렬 계산
        # 공식: Z = (X - Mean) / Std
        z_matrix = (dense_tfidf - means) / stds
        
        # 6. 결과 포매팅 (Matrix -> List of Dicts)
        results = []
        for i in range(len(docs_subset)):
            doc_scores = {}
            # 해당 문서에서 등장한 단어(0이 아닌 값)만 필터링하여 저장 (저장 용량 최적화)
            nonzero_indices = dense_tfidf[i].nonzero()[0]
            
            for idx in nonzero_indices:
                word = feature_names[idx]
                score = z_matrix[i][idx]
                # 소수점 4자리까지 반올림하여 저장
                doc_scores[word] = round(float(score), 4)
                
            results.append(doc_scores)
            
        return results

    def run(self):
        """
        [Execution Flow]
        전체 파이프라인을 실행합니다.
        1. 데이터 로드
        2. Global Z-Score 계산 (전체 문서 대상)
        3. Local Z-Score 계산 (도메인별 그룹핑 후 대상)
        4. 결과 파일 저장
        """
        self.load_all_data()
        
        if not self.documents:
            logger.warning("[ZScore] No documents to process. Terminating.")
            return

        # --- Step 1: Global Analysis ---
        logger.info("[ZScore] Calculating Global Z-scores...")
        # 전체 문서를 대상으로 통계 산출
        global_scores = self._compute_stats(self.documents)
        
        # 계산 결과를 메모리상의 문서 객체에 임시 저장
        for i, doc in enumerate(self.documents):
            doc['global_z'] = global_scores[i]

        # --- Step 2: Local (Domain) Analysis ---
        logger.info("[ZScore] Calculating Local (Domain) Z-scores...")
        
        # 도메인별로 문서를 그룹핑
        domain_groups = defaultdict(list)
        doc_indices_map = defaultdict(list) # 결과를 다시 원본 리스트에 매핑하기 위한 인덱스 저장소

        for i, doc in enumerate(self.documents):
            domain_groups[doc['domain_dir']].append(doc)
            doc_indices_map[doc['domain_dir']].append(i)

        # 도메인 그룹별로 루프를 돌며 계산
        for domain_dir, group_docs in domain_groups.items():
            local_scores = self._compute_stats(group_docs)
            
            # 계산된 로컬 점수를 원본 문서 객체에 매핑
            for local_idx, score_dict in enumerate(local_scores):
                original_idx = doc_indices_map[domain_dir][local_idx]
                self.documents[original_idx]['local_z'] = score_dict

        # --- Step 3: Save Results ---
        self._save_results()

    def _save_results(self):
        """
        [File Saver]
        계산된 Global/Local 점수를 각 도메인 폴더 내 'z_score.json' 파일로 저장합니다.
        """
        logger.info("[ZScore] Saving results to JSON files...")
        
        # 데이터를 도메인별로 다시 정리
        # 구조: results_by_domain[domain_name][doc_id] = {global: ..., local: ...}
        results_by_domain = defaultdict(dict)
        
        for doc in self.documents:
            results_by_domain[doc['domain_dir']][doc['doc_id']] = {
                "global": doc.get('global_z', {}),
                "local": doc.get('local_z', {})
            }
            
        # 도메인별로 파일 쓰기 수행
        saved_count = 0
        for domain_dir, data_dict in results_by_domain.items():
            save_path = os.path.join(self.data_root, domain_dir, "z_score.json")
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(data_dict, f, ensure_ascii=False, indent=2)
                saved_count += 1
            except Exception as e:
                logger.error(f"[ZScore] Failed to save {save_path}: {e}")
                
        logger.info(f"[ZScore] Successfully updated z_score.json for {saved_count} domains.")