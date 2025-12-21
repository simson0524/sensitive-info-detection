# src/modules/z_score_calculator.py

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Set, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from transformers import AutoTokenizer

# [Logger Setup]
# src.utils.logger 모듈에서 설정 함수를 가져옵니다.
# 클래스 내부에서 독립적인 로거 인스턴스를 생성하기 위함입니다.
from src.utils.logger import setup_experiment_logger 

class ZScoreCalculator:
    """
    [Z-Score Calculator]
    문서 집합에서 TF-IDF를 기반으로 각 토큰(Subword)의 통계적 중요도(Z-Score)를 계산하는 클래스입니다.
    
    [핵심 특징]
    1. Model-Aligned Tokenization:
       - 한국어 형태소 분석기(Okt) 대신 학습 모델(RoBERTa 등)의 Tokenizer를 직접 사용합니다.
       - 모델이 실제로 학습하게 될 Input Feature(Subword) 단위로 통계를 산출하여 정합성을 높입니다.
       
    2. Hybrid Scoring Strategy:
       - Case A (정답지 있음): 정답지 단어를 Subword로 쪼개어 해당 토큰들의 점수를 추적합니다. (점수가 낮아도 기록)
       - Case B (정답지 없음): 문서 내 등장하는 모든 유효 토큰에 대해 점수를 계산합니다.
    """

    def __init__(self, data_root_dir: str = 'data/train_data', model_name: str = "klue/roberta-base"):
        """
        Args:
            data_root_dir (str): 데이터가 위치한 최상위 디렉토리 (기본: data/train_data)
            model_name (str): 사용할 HuggingFace 모델 이름 (기본: klue/roberta-base)
        """
        # 1. 로거 설정 (실험 코드: Z_SCORE_CALC)
        # 로그 파일 위치: outputs/logs/Z_SCORE_CALC/Z_SCORE_CALC_experiment_log.txt
        self.logger = setup_experiment_logger(experiment_code="Z_SCORE_CALC")

        self.data_root = data_root_dir
        self.documents: List[Dict[str, Any]] = [] 
        
        # 도메인별 타겟 단어장 (Case A: Set[str], Case B: None)
        self.domain_target_vocab: Dict[str, Optional[Set[str]]] = {}
        
        # 2. 토크나이저 로드
        self.logger.info(f"[ZScore] Loading tokenizer from: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer '{model_name}': {e}")
            raise e
        
        # 3. 특수 토큰 필터링 준비
        # [CLS], [SEP], [PAD] 등은 통계적 분석에서 제외하기 위해 미리 식별합니다.
        self.special_tokens = set(self.tokenizer.all_special_tokens)

    def _smart_tokenizer(self, text: str) -> List[str]:
        """
        [토크나이저 래퍼]
        TfidfVectorizer가 사용할 커스텀 토크나이저 함수입니다.
        텍스트를 모델의 Subword 단위로 분리하고, 특수 토큰을 제거합니다.
        
        Ex) "정보보호" -> ['정보', '##보호']
        """
        if not isinstance(text, str):
            return []
        
        # HuggingFace Tokenizer의 tokenize 메서드 사용
        tokens = self.tokenizer.tokenize(text)
        
        # 특수 토큰 제거 (순수 텍스트 토큰만 남김)
        valid_tokens = [t for t in tokens if t not in self.special_tokens]
        
        return valid_tokens

    def load_all_data(self):
        """
        [데이터 로드 프로세스]
        1. 도메인 디렉토리 순회
        2. answer_sheet.csv가 있다면 해당 단어들을 모델 토크나이저로 분해하여 Target Vocab 구축
        3. 각 JSON 문서를 읽어 메모리에 적재
        """
        abs_root = os.path.abspath(self.data_root)
        self.logger.info(f"[ZScore] Scanning data root: {abs_root}")
        
        if not os.path.exists(abs_root):
            self.logger.error(f"Data root path does not exist: {abs_root}")
            return

        self.documents = []
        self.domain_target_vocab = {}

        for domain_dir in os.listdir(abs_root):
            domain_path = os.path.join(abs_root, domain_dir)
            
            # 숨김 파일이나 파일은 건너뛰기
            if not os.path.isdir(domain_path) or domain_dir.startswith('.'):
                continue
            
            # --- [Part 1] 정답지(CSV) 처리 ---
            answer_sheet_path = os.path.join(domain_path, 'answer_sheet.csv')
            target_vocab = None
            
            if os.path.exists(answer_sheet_path):
                try:
                    df = pd.read_csv(answer_sheet_path)
                    # 컬럼명 호환성 처리 (word 혹은 첫 번째 컬럼)
                    if 'word' in df.columns:
                        col_data = df['word']
                    else:
                        col_data = df.iloc[:, 0]
                    
                    raw_words = col_data.dropna().astype(str).tolist()
                    processed_vocab = set()
                    
                    # [핵심] 정답지에 있는 단어도 '모델 토크나이저'로 쪼개서 등록해야 합니다.
                    # 그래야 나중에 문서 본문(JSON)을 분석한 결과와 Key가 매칭됩니다.
                    # Ex: CSV "삼성전자" -> {'삼성', '##전자'} 등록
                    for w in raw_words:
                        tokens = self._smart_tokenizer(w)
                        processed_vocab.update(tokens)
                        
                    target_vocab = processed_vocab
                    self.logger.info(f"[ZScore] Domain '{domain_dir}': Loaded {len(target_vocab)} subword tokens from CSV.")
                    
                except Exception as e:
                    self.logger.warning(f"[ZScore] CSV Error in {domain_dir}: {e}")
            else:
                self.logger.info(f"[ZScore] Domain '{domain_dir}': No CSV. Using auto-extraction mode.")
            
            self.domain_target_vocab[domain_dir] = target_vocab

            # --- [Part 2] JSON 문서 로드 ---
            for file_name in os.listdir(domain_path):
                if not file_name.endswith('.json') or file_name == 'z_score.json':
                    continue
                
                file_path = os.path.join(domain_path, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 문서 내 모든 문장을 공백으로 연결
                        full_text = " ".join([item.get('sentence', '') for item in data])
                        
                        self.documents.append({
                            'doc_id': file_name.replace('.json', ''),
                            'domain_dir': domain_dir, 
                            'full_text': full_text
                        })
                except Exception:
                    pass

    def _compute_stats(self, docs_subset: List[Dict], vocab: Optional[Set[str]] = None) -> List[Dict[str, float]]:
        """
        [통계 계산 엔진]
        TF-IDF 행렬 생성 -> 평균/표준편차 계산 -> Z-Score 산출
        """
        if not docs_subset:
            return []

        corpus = [d['full_text'] for d in docs_subset]
        
        # TfidfVectorizer에 모델 토크나이저 주입
        vectorizer = TfidfVectorizer(
            tokenizer=self._smart_tokenizer,
            token_pattern=None, # tokenizer 파라미터 사용 시 None 필수
            vocabulary=vocab    # vocab이 None이면 전체 토큰 자동 추출, Set이면 해당 토큰만 계산
        )
        
        try:
            # TF-IDF 행렬 생성 (행: 문서, 열: 단어)
            tfidf_matrix = vectorizer.fit_transform(corpus)
        except ValueError:
            # 단어가 하나도 없는 경우 등
            return [{} for _ in docs_subset]

        feature_names = vectorizer.get_feature_names_out()
        dense_tfidf = tfidf_matrix.toarray()
        
        # 1. 평균(Mean) 및 표준편차(Std) 계산
        means = np.mean(dense_tfidf, axis=0)
        stds = np.std(dense_tfidf, axis=0)
        
        # Zero Division 방지 (표준편차가 0인 경우 1.0으로 대체)
        stds[stds == 0] = 1.0 
        
        # 2. Z-Score 계산
        z_matrix = (dense_tfidf - means) / stds
        
        results = []
        for i in range(len(docs_subset)):
            doc_scores = {}
            
            if vocab is not None:
                # [Case A] 정답지가 지정된 경우
                # 문서에 등장하지 않아 점수가 0이더라도 모든 타겟 단어의 점수를 기록합니다.
                # (분석 시 "점수가 낮음"을 확인하기 위함)
                for idx, word in enumerate(feature_names):
                    score = z_matrix[i][idx]
                    doc_scores[word] = round(float(score), 4)
            else:
                # [Case B] 전체 자동 추출인 경우
                # 데이터 양을 줄이기 위해 실제 등장한(0이 아닌) 단어만 저장합니다.
                nonzero_indices = dense_tfidf[i].nonzero()[0]
                for idx in nonzero_indices:
                    word = feature_names[idx]
                    score = z_matrix[i][idx]
                    doc_scores[word] = round(float(score), 4)
            
            results.append(doc_scores)
            
        return results

    def run(self):
        """
        [실행 파이프라인]
        Load Data -> Calculate Global Z -> Calculate Local Z -> Save Results
        """
        self.load_all_data()
        
        if not self.documents:
            self.logger.warning("No documents found to process.")
            return

        # --- Step 1: Global Z-Score (전체 데이터 기준) ---
        self.logger.info("[ZScore] Calculating Global Z-scores...")
        # 전체 문서에 대해 모든 토큰 통계 산출 (vocab=None)
        global_scores_full = self._compute_stats(self.documents, vocab=None)
        
        for i, doc in enumerate(self.documents):
            target_vocab = self.domain_target_vocab.get(doc['domain_dir'])
            
            if target_vocab:
                # [필터링] 정답지가 있는 경우, 전체 통계 중에서 정답지 단어만 추출
                filtered = {}
                for target_token in target_vocab:
                    # 키가 없으면(문서 미등장) 0.0으로 처리하여 누락 방지
                    score = global_scores_full[i].get(target_token, 0.0)
                    filtered[target_token] = score
                doc['global_z'] = filtered
            else:
                # 정답지가 없으면 전체 저장
                doc['global_z'] = global_scores_full[i]

        # --- Step 2: Local Z-Score (도메인별 기준) ---
        self.logger.info("[ZScore] Calculating Local Z-scores...")
        domain_groups = defaultdict(list)
        doc_indices_map = defaultdict(list)

        # 도메인별 그룹핑
        for i, doc in enumerate(self.documents):
            domain_groups[doc['domain_dir']].append(doc)
            doc_indices_map[doc['domain_dir']].append(i)

        for domain_dir, group_docs in domain_groups.items():
            target_vocab = self.domain_target_vocab.get(domain_dir)
            # 도메인 그룹 내에서 다시 통계 계산
            local_scores = self._compute_stats(group_docs, vocab=target_vocab)
            
            # 원본 문서 객체에 결과 매핑
            for local_idx, score_dict in enumerate(local_scores):
                original_idx = doc_indices_map[domain_dir][local_idx]
                self.documents[original_idx]['local_z'] = score_dict

        # --- Step 3: 저장 ---
        self._save_results()

    def _save_results(self):
        """
        도메인별 z_score.json 파일 저장
        """
        self.logger.info("[ZScore] Saving results...")
        results_by_domain = defaultdict(dict)
        
        for doc in self.documents:
            results_by_domain[doc['domain_dir']][doc['doc_id']] = {
                "global": doc.get('global_z', {}),
                "local": doc.get('local_z', {})
            }
        
        for domain_dir, data_dict in results_by_domain.items():
            save_path = os.path.join(self.data_root, domain_dir, "z_score.json")
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(data_dict, f, ensure_ascii=False, indent=2)
            except Exception as e:
                self.logger.error(f"Failed to save {save_path}: {e}")