# src/modules/z_score_calculator.py

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Set, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from src.utils.logger import logger

class ZScoreCalculator:
    """
    [Z-Score Calculator Module]
    
    데이터셋 내의 문서들에 대해 TF-IDF 기반 Z-Score(중요도 점수)를 계산하여,
    각 단어가 통계적으로 얼마나 유의미한지 수치화합니다.
    
    [핵심 기능]
    1. Global Z-Score: 전체 도메인 통합 통계 기반 (전역적 중요도)
       - 계산은 전체 단어 대상으로 수행하여 정확한 IDF를 산출한 뒤, 저장 시 필터링합니다.
    2. Local Z-Score: 특정 도메인 내부 통계 기반 (지역적 중요도)
       - 해당 도메인 내에서의 상대적 중요도를 파악합니다.
    3. Target Vocabulary Filtering: 
       - 도메인 내에 'answer_sheet.csv'가 존재할 경우, 'word' 컬럼에 명시된 단어들에 대해서만
         선택적으로 점수를 남기고 나머지는 제거합니다. (노이즈 제거 및 용량 최적화)
    """

    def __init__(self, data_root_dir: str = 'data/train_data'):
        """
        Args:
            data_root_dir (str): 데이터가 위치한 최상위 디렉토리 경로 (예: 'data/train_data')
        """
        self.data_root = data_root_dir
        
        # self.documents: 로드된 모든 문서 정보를 담는 리스트
        # 구조: [{'doc_id': str, 'domain_dir': str, 'full_text': str, ...}, ...]
        self.documents: List[Dict[str, Any]] = [] 
        
        # self.domain_target_vocab: 도메인별 타겟 단어장 (Set 구조로 빠른 검색 지원)
        # 구조: {'medical': {'수술', '환자', ...}, 'law': None, ...}
        self.domain_target_vocab: Dict[str, Optional[Set[str]]] = {}

    def load_all_data(self):
        """
        [Data Loader]
        지정된 루트 디렉토리 하위의 모든 도메인 폴더를 순회합니다.
        1. 'answer_sheet.csv'가 있다면 로드하여 타겟 단어장(Target Vocabulary)을 구축합니다.
        2. '.json' 문서 파일을 로드하고, 문장들을 합쳐 TF-IDF용 텍스트로 변환합니다.
        """
        # 경로 안전성을 위해 절대 경로로 변환
        abs_root = os.path.abspath(self.data_root)
        logger.info(f"[ZScore] Scanning data root: {abs_root}")
        
        if not os.path.exists(abs_root):
            logger.error(f"[ZScore] Data root not found: {abs_root}")
            return

        # 초기화
        self.documents = []
        self.domain_target_vocab = {}

        # 1. 도메인 폴더 순회 (1-depth)
        for domain_dir in os.listdir(abs_root):
            domain_path = os.path.join(abs_root, domain_dir)
            
            # 디렉토리가 아니거나 숨김 파일(.git, .DS_Store 등)은 패스
            if not os.path.isdir(domain_path) or domain_dir.startswith('.'):
                continue
            
            # --- [Step A] Target Vocabulary Load (answer_sheet.csv) ---
            answer_sheet_path = os.path.join(domain_path, 'answer_sheet.csv')
            target_vocab = None
            
            if os.path.exists(answer_sheet_path):
                try:
                    # Pandas로 CSV 로드
                    df = pd.read_csv(answer_sheet_path)
                    
                    # 'word' 컬럼이 있으면 사용, 없으면 첫 번째 컬럼 사용 (유연한 처리)
                    if 'word' in df.columns:
                        col_data = df['word']
                    else:
                        col_data = df.iloc[:, 0]
                        logger.warning(f"[ZScore] Domain '{domain_dir}': 'word' column missing in answer_sheet.csv. Using first column.")
                    
                    # 문자열 변환 -> 공백 제거 -> Set 변환 (중복 제거)
                    words = col_data.dropna().astype(str).str.strip().tolist()
                    target_vocab = set(words)
                    
                    logger.info(f"[ZScore] Domain '{domain_dir}': Loaded {len(target_vocab)} target words from answer_sheet.csv.")
                    
                except Exception as e:
                    logger.warning(f"[ZScore] Failed to read answer_sheet.csv in {domain_dir}: {e}")
            
            self.domain_target_vocab[domain_dir] = target_vocab

            # --- [Step B] Document JSON Load ---
            for file_name in os.listdir(domain_path):
                # .json 파일만 처리하되, 결과 파일(z_score.json)은 제외
                if not file_name.endswith('.json') or file_name == 'z_score.json':
                    continue
                
                file_path = os.path.join(domain_path, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # 문서 내 모든 문장("sentence")을 공백으로 이어 붙여 하나의 텍스트로 만듦
                        # (TF-IDF Vectorizer 입력용)
                        full_text = " ".join([item.get('sentence', '') for item in data])
                        
                        self.documents.append({
                            'doc_id': file_name.replace('.json', ''),
                            'domain_dir': domain_dir, 
                            'full_text': full_text
                        })
                except Exception as e:
                    logger.warning(f"[ZScore] Failed to load {file_name}: {e}")

        logger.info(f"[ZScore] Total loaded documents: {len(self.documents)}")

    def _compute_stats(self, docs_subset: List[Dict], vocab: Optional[Set[str]] = None) -> List[Dict[str, float]]:
        """
        [Core Logic] TF-IDF & Z-Score 계산
        
        Args:
            docs_subset (List[Dict]): 계산 대상 문서 리스트
            vocab (Optional[Set[str]]): 
                - None: 전체 단어를 대상으로 계산합니다.
                - Set: 지정된 단어들에 대해서만 행렬을 생성하고 계산합니다.
        
        Returns:
            List[Dict[str, float]]: 각 문서별 {단어: z_score} 딕셔너리 리스트
        """
        if not docs_subset:
            return []

        corpus = [d['full_text'] for d in docs_subset]
        
        # 1. TF-IDF Vectorizer 설정
        # token_pattern: 단어 경계(\b) 기준, 한글/영어/숫자 포함(\w+)
        vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b",
            vocabulary=vocab # vocab이 있으면 해당 단어만 벡터화 (속도 향상 & 필터링)
        )
        
        try:
            # TF-IDF 행렬 생성 (Sparse Matrix)
            tfidf_matrix = vectorizer.fit_transform(corpus)
        except ValueError:
            # vocab에 있는 단어가 코퍼스에 하나도 없거나 문서가 비어있는 경우 예외 처리
            return [{} for _ in docs_subset]

        feature_names = vectorizer.get_feature_names_out()
        dense_tfidf = tfidf_matrix.toarray()
        
        # 2. 통계 계산 (Column-wise: 단어별)
        # Global 계산 시 0을 포함해야 전체 문서 대비 위치를 알 수 있음
        means = np.mean(dense_tfidf, axis=0)
        stds = np.std(dense_tfidf, axis=0)
        
        # [Zero Division Handling]
        # 표준편차가 0인 경우(모든 문서에서 값이 같음) 1.0으로 대체하여 NaN 방지
        stds[stds == 0] = 1.0 
        
        # 3. Z-Score 산출: (값 - 평균) / 표준편차
        z_matrix = (dense_tfidf - means) / stds
        
        # 4. 결과 포매팅
        results = []
        for i in range(len(docs_subset)):
            doc_scores = {}
            # 0이 아닌 값(등장한 단어)에 대해서만 저장
            nonzero_indices = dense_tfidf[i].nonzero()[0]
            
            for idx in nonzero_indices:
                word = feature_names[idx]
                score = z_matrix[i][idx]
                doc_scores[word] = round(float(score), 4) # 소수점 4자리 반올림
            results.append(doc_scores)
            
        return results

    def run(self):
        """
        [Execution Flow]
        전체 파이프라인을 실행합니다.
        1. 데이터 로드 (JSON Docs & Answer Sheets)
        2. Global Z-Score 계산 (전체 단어 기준 계산 후 필터링)
        3. Local Z-Score 계산 (타겟 단어 기준 계산)
        4. 결과 파일 저장
        """
        self.load_all_data()
        
        if not self.documents:
            logger.warning("[ZScore] No documents to process. Terminating.")
            return

        # --- [Step 1] Global Analysis ---
        logger.info("[ZScore] Calculating Global Z-scores...")
        
        # 주의: Global 통계의 정확성(IDF)을 위해 일단 '전체 단어(vocab=None)'로 계산합니다.
        # 특정 단어만 미리 필터링해서 계산하면 전체 말뭉치 내의 희소성 정보가 왜곡될 수 있습니다.
        global_scores_full = self._compute_stats(self.documents, vocab=None)
        
        # 계산 후 저장 단계에서 필터링 적용
        for i, doc in enumerate(self.documents):
            target_vocab = self.domain_target_vocab.get(doc['domain_dir'])
            
            if target_vocab:
                # [Filter] 해당 도메인의 answer_sheet에 있는 단어만 남김
                filtered_score = {k: v for k, v in global_scores_full[i].items() if k in target_vocab}
                doc['global_z'] = filtered_score
            else:
                # answer_sheet가 없으면 전체 저장
                doc['global_z'] = global_scores_full[i]

        # --- [Step 2] Local (Domain) Analysis ---
        logger.info("[ZScore] Calculating Local (Domain) Z-scores...")
        
        # 도메인별로 문서 그룹핑
        domain_groups = defaultdict(list)
        doc_indices_map = defaultdict(list)

        for i, doc in enumerate(self.documents):
            domain_groups[doc['domain_dir']].append(doc)
            doc_indices_map[doc['domain_dir']].append(i)

        for domain_dir, group_docs in domain_groups.items():
            target_vocab = self.domain_target_vocab.get(domain_dir)
            
            if target_vocab:
                # [Optimization] Local 계산 시에는 처음부터 타겟 단어만 계산해도 무방함
                # (해당 도메인 내에서의 분포만 중요하므로)
                local_scores = self._compute_stats(group_docs, vocab=target_vocab)
            else:
                local_scores = self._compute_stats(group_docs, vocab=None)
            
            # 계산 결과를 원본 문서 객체에 매핑
            for local_idx, score_dict in enumerate(local_scores):
                original_idx = doc_indices_map[domain_dir][local_idx]
                self.documents[original_idx]['local_z'] = score_dict

        # --- [Step 3] Save Results ---
        self._save_results()

    def _save_results(self):
        """
        [File Saver]
        계산된 결과를 각 도메인 디렉토리에 'z_score.json' 파일로 저장합니다.
        """
        logger.info("[ZScore] Saving results to JSON files...")
        
        # 도메인별 데이터 구조화
        # 구조: { "doc_id": { "global": {...}, "local": {...} } }
        results_by_domain = defaultdict(dict)
        
        for doc in self.documents:
            results_by_domain[doc['domain_dir']][doc['doc_id']] = {
                "global": doc.get('global_z', {}),
                "local": doc.get('local_z', {})
            }
            
        # 파일 쓰기
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