# src/modules/z_score_calculator.py

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Set, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from src.utils.logger import logger

# [변경] KoNLPy 제거 -> Transformers 추가
from transformers import AutoTokenizer

class ZScoreCalculator:
    """
    [Z-Score Calculator]
    문서 집합에서 TF-IDF를 기반으로 토큰(Subword)의 통계적 중요도(Z-Score)를 계산합니다.
    Deep Learning 모델(RoBERTa 등)의 토크나이저를 직접 사용하여,
    모델이 학습하는 Input Feature 단위의 정확한 통계를 산출합니다.
    """

    def __init__(self, data_root_dir: str = 'data/train_data', model_name: str = "klue/roberta-base"):
        self.data_root = data_root_dir
        self.documents: List[Dict[str, Any]] = [] 
        self.domain_target_vocab: Dict[str, Optional[Set[str]]] = {}
        
        # [핵심 변경] 모델의 토크나이저 로드
        logger.info(f"[ZScore] Loading tokenizer from: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Failed to load tokenizer '{model_name}': {e}")
            raise e
        
        # 분석에서 제외할 특수 토큰 집합 ([CLS], [SEP], [PAD] 등)
        self.special_tokens = set(self.tokenizer.all_special_tokens)

    def _smart_tokenizer(self, text: str) -> List[str]:
        """
        [Tokenizer Wrapper]
        텍스트를 모델의 Subword 단위로 토큰화합니다.
        예: "정보보호" -> ['정보', '##보호']
        """
        if not isinstance(text, str):
            return []
        
        # 1. 모델 토크나이저 실행
        tokens = self.tokenizer.tokenize(text)
        
        # 2. 특수 토큰 제거 (통계적 의미가 없는 토큰 필터링)
        valid_tokens = [t for t in tokens if t not in self.special_tokens]
        
        return valid_tokens

    def load_all_data(self):
        """
        데이터 로드 및 정답지(CSV) 전처리
        """
        abs_root = os.path.abspath(self.data_root)
        logger.info(f"[ZScore] Scanning data root: {abs_root}")
        
        if not os.path.exists(abs_root):
            return

        self.documents = []
        self.domain_target_vocab = {}

        for domain_dir in os.listdir(abs_root):
            domain_path = os.path.join(abs_root, domain_dir)
            if not os.path.isdir(domain_path) or domain_dir.startswith('.'):
                continue
            
            # --- [Case A] answer_sheet.csv 처리 ---
            answer_sheet_path = os.path.join(domain_path, 'answer_sheet.csv')
            target_vocab = None
            
            if os.path.exists(answer_sheet_path):
                try:
                    df = pd.read_csv(answer_sheet_path)
                    if 'word' in df.columns:
                        col_data = df['word']
                    else:
                        col_data = df.iloc[:, 0]
                    
                    raw_words = col_data.dropna().astype(str).tolist()
                    processed_vocab = set()
                    
                    # [핵심 로직] 정답지 단어도 '모델 토크나이저'로 쪼개서 등록
                    # 예: CSV "삼성전자" -> {'삼성', '##전자'} 등록
                    for w in raw_words:
                        tokens = self._smart_tokenizer(w)
                        processed_vocab.update(tokens)
                        
                    target_vocab = processed_vocab
                    logger.info(f"[ZScore] Domain '{domain_dir}': Loaded {len(target_vocab)} subword tokens from CSV.")
                    
                except Exception as e:
                    logger.warning(f"[ZScore] CSV Error in {domain_dir}: {e}")
            else:
                logger.info(f"[ZScore] Domain '{domain_dir}': No CSV. Using auto-extraction mode.")
            
            self.domain_target_vocab[domain_dir] = target_vocab

            # --- [문서 파일 로드] ---
            for file_name in os.listdir(domain_path):
                if not file_name.endswith('.json') or file_name == 'z_score.json':
                    continue
                
                file_path = os.path.join(domain_path, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        full_text = " ".join([item.get('sentence', '') for item in data])
                        
                        self.documents.append({
                            'doc_id': file_name.replace('.json', ''),
                            'domain_dir': domain_dir, 
                            'full_text': full_text
                        })
                except Exception:
                    pass

    def _compute_stats(self, docs_subset: List[Dict], vocab: Optional[Set[str]] = None) -> List[Dict[str, float]]:
        if not docs_subset:
            return []

        corpus = [d['full_text'] for d in docs_subset]
        
        # TfidfVectorizer에 모델 토크나이저 주입
        vectorizer = TfidfVectorizer(
            tokenizer=self._smart_tokenizer,
            token_pattern=None,
            vocabulary=vocab
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
        except ValueError:
            return [{} for _ in docs_subset]

        feature_names = vectorizer.get_feature_names_out()
        dense_tfidf = tfidf_matrix.toarray()
        
        means = np.mean(dense_tfidf, axis=0)
        stds = np.std(dense_tfidf, axis=0)
        stds[stds == 0] = 1.0 
        
        z_matrix = (dense_tfidf - means) / stds
        
        results = []
        for i in range(len(docs_subset)):
            doc_scores = {}
            
            # [안전장치 적용] 
            # vocab이 지정된 경우(정답지 모드)에는 점수가 0이어도 모두 기록합니다.
            if vocab is not None:
                for idx, word in enumerate(feature_names):
                    score = z_matrix[i][idx]
                    doc_scores[word] = round(float(score), 4)
            else:
                # 전체 모드에서는 너무 많으니 0이 아닌 것만 기록
                nonzero_indices = dense_tfidf[i].nonzero()[0]
                for idx in nonzero_indices:
                    word = feature_names[idx]
                    score = z_matrix[i][idx]
                    doc_scores[word] = round(float(score), 4)
            
            results.append(doc_scores)
            
        return results

    def run(self):
        self.load_all_data()
        
        if not self.documents:
            return

        logger.info("[ZScore] Calculating Global Z-scores...")
        global_scores_full = self._compute_stats(self.documents, vocab=None)
        
        for i, doc in enumerate(self.documents):
            target_vocab = self.domain_target_vocab.get(doc['domain_dir'])
            
            if target_vocab:
                # [Case A] 정답지 단어는 점수가 0이어도 강제로 가져옴 (누락 방지)
                filtered = {}
                for target_token in target_vocab:
                    score = global_scores_full[i].get(target_token, 0.0)
                    filtered[target_token] = score
                doc['global_z'] = filtered
            else:
                doc['global_z'] = global_scores_full[i]

        logger.info("[ZScore] Calculating Local Z-scores...")
        domain_groups = defaultdict(list)
        doc_indices_map = defaultdict(list)

        for i, doc in enumerate(self.documents):
            domain_groups[doc['domain_dir']].append(doc)
            doc_indices_map[doc['domain_dir']].append(i)

        for domain_dir, group_docs in domain_groups.items():
            target_vocab = self.domain_target_vocab.get(domain_dir)
            local_scores = self._compute_stats(group_docs, vocab=target_vocab)
            
            for local_idx, score_dict in enumerate(local_scores):
                original_idx = doc_indices_map[domain_dir][local_idx]
                self.documents[original_idx]['local_z'] = score_dict

        self._save_results()

    def _save_results(self):
        logger.info("[ZScore] Saving results...")
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
                logger.error(f"Failed to save {save_path}: {e}")