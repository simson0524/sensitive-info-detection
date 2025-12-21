# src/modules/z_score_calculator.py

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Set, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from transformers import AutoTokenizer

# [변경] Sudo 권한 없이 설치 가능한 Mecab 라이브러리
from mecab import MeCab 
from src.utils.logger import setup_experiment_logger 

class ZScoreCalculator:
    """
    [Z-Score Calculator]
    BERT Tokenizer의 '재결합(Reconstruction)' 능력과 MeCab의 '형태소 분석(POS Tagging)' 능력을 결합하여,
    문맥상 의미 있는 알맹이 단어(명사, 숫자, 외국어 등)의 통계적 중요도(Z-Score)를 계산합니다.
    """

    def __init__(self, data_root_dir: str = 'data/train_data', model_name: str = "klue/roberta-base"):
        # 1. 독립적인 로거 생성 (로그 경로: outputs/logs/Z_SCORE_CALC)
        self.logger = setup_experiment_logger(experiment_code="Z_SCORE_CALC")
        
        self.data_root = data_root_dir
        self.documents: List[Dict[str, Any]] = [] 
        self.domain_target_vocab: Dict[str, Optional[Set[str]]] = {}
        
        # 2. BERT Tokenizer 로드 (Subword 분해용)
        self.logger.info(f"[ZScore] Loading tokenizer from: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer '{model_name}': {e}")
            raise e
        
        self.special_tokens = set(self.tokenizer.all_special_tokens)
        
        # 3. MeCab 초기화 (python-mecab-ko 사용)
        self.logger.info("[ZScore] Initializing MeCab (python-mecab-ko)...")
        try:
            self.mecab = MeCab()
        except Exception as e:
            self.logger.critical("Failed to initialize MeCab. Please run 'pip install python-mecab-ko'")
            raise e
        
        # [최적화] 형태소 분석 결과 캐시 (메모이제이션)
        # 반복되는 단어(예: '삼성전자')의 중복 분석을 방지하여 속도를 비약적으로 향상시킵니다.
        self.pos_cache: Dict[str, List[str]] = {}

    def _smart_tokenizer(self, text: str) -> List[str]:
        """
        [Advanced Hybrid Tokenizer]
        1. Tokenize: BERT 모델 기준으로 문장을 쪼갭니다.
        2. Merge: '##'으로 시작하는 서브워드를 앞 단어와 합쳐 온전한 어절로 복원합니다.
        3. Filter: 복원된 어절을 MeCab으로 분석해 조사 등을 제거하고 핵심 품사만 남깁니다.
        
        Returns:
            List[str]: 정제된 핵심 단어 리스트
        """
        if not isinstance(text, str):
            return []
        
        # --- Step 1: RoBERTa Tokenization ---
        # 예: "삼성전자가" -> ['삼성', '##전자', '##가']
        raw_tokens = self.tokenizer.tokenize(text)
        merged_chunks = []
        
        # --- Step 2: Merge Subwords (Reconstruction) ---
        # 예: ['삼성', '##전자', '##가'] -> ['삼성전자가']
        for t in raw_tokens:
            if t in self.special_tokens: continue
            
            if t.startswith("##"):
                if merged_chunks:
                    merged_chunks[-1] += t[2:]
                else:
                    merged_chunks.append(t[2:])
            else:
                merged_chunks.append(t)
        
        # --- Step 3: MeCab POS Filtering (with Cache) ---
        final_tokens = []
        
        # 추출할 핵심 품사 (MeCab 태그 기준)
        # NNG:일반명사, NNP:고유명사, NNB:의존명사, NR:수사, SL:외국어, SN:숫자
        TARGET_TAGS = {'NNG', 'NNP', 'NNB', 'NR', 'SL', 'SN'} 
        
        for chunk in merged_chunks:
            # 3-1. 캐시 확인 (이미 분석한 단어면 바로 반환)
            if chunk in self.pos_cache:
                final_tokens.extend(self.pos_cache[chunk])
                continue

            # 3-2. 캐시에 없으면 분석 수행
            try:
                valid_words = []
                pos_results = self.mecab.pos(chunk)
                
                for word, tag in pos_results:
                    # 태그가 타겟 품사에 속하는지 확인
                    if tag in TARGET_TAGS:
                        valid_words.append(word)
                
                # 3-3. 결과 캐싱
                self.pos_cache[chunk] = valid_words
                final_tokens.extend(valid_words)
                
            except Exception:
                pass
                
        return final_tokens

    def load_all_data(self):
        """
        데이터 로드 및 정답지(CSV) 전처리 프로세스
        """
        abs_root = os.path.abspath(self.data_root)
        self.logger.info(f"[ZScore] Scanning data root: {abs_root}")
        
        if not os.path.exists(abs_root):
            return

        self.documents = []
        self.domain_target_vocab = {}

        for domain_dir in os.listdir(abs_root):
            domain_path = os.path.join(abs_root, domain_dir)
            if not os.path.isdir(domain_path) or domain_dir.startswith('.'):
                continue
            
            # --- Answer Sheet CSV 처리 ---
            answer_sheet_path = os.path.join(domain_path, 'answer_sheet.csv')
            target_vocab = None
            
            if os.path.exists(answer_sheet_path):
                try:
                    df = pd.read_csv(answer_sheet_path)
                    # 컬럼명 호환성 체크
                    if 'word' in df.columns: col_data = df['word']
                    else: col_data = df.iloc[:, 0]
                    
                    raw_words = col_data.dropna().astype(str).tolist()
                    processed_vocab = set()
                    
                    # [중요] 정답지 단어도 동일한 로직(Tokenize->Merge->Filter)을 거쳐야
                    # 문서에서 분석된 단어와 Key가 일치하게 됩니다.
                    for w in raw_words:
                        tokens = self._smart_tokenizer(w)
                        processed_vocab.update(tokens)
                        
                    target_vocab = processed_vocab
                    self.logger.info(f"[ZScore] Domain '{domain_dir}': Loaded {len(target_vocab)} refined words from CSV.")
                    
                except Exception as e:
                    self.logger.warning(f"[ZScore] CSV Error in {domain_dir}: {e}")
            else:
                self.logger.info(f"[ZScore] Domain '{domain_dir}': No CSV. Using auto-extraction mode.")
            
            self.domain_target_vocab[domain_dir] = target_vocab

            # --- JSON 문서 로드 ---
            for file_name in os.listdir(domain_path):
                # z_score.json 파일은 제외
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
        """
        TF-IDF 및 Z-Score 계산 엔진
        """
        if not docs_subset:
            return []

        corpus = [d['full_text'] for d in docs_subset]
        
        # [중요] lowercase=False: 대소문자 구분을 유지하여 고유명사(DNA, CEO 등)를 정확히 식별
        vectorizer = TfidfVectorizer(
            tokenizer=self._smart_tokenizer,
            token_pattern=None,
            vocabulary=vocab,
            lowercase=False 
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
        except ValueError:
            return [{} for _ in docs_subset]

        feature_names = vectorizer.get_feature_names_out()
        dense_tfidf = tfidf_matrix.toarray()
        
        means = np.mean(dense_tfidf, axis=0)
        stds = np.std(dense_tfidf, axis=0)
        stds[stds == 0] = 1.0 # Zero Division 방지
        
        z_matrix = (dense_tfidf - means) / stds
        
        results = []
        for i in range(len(docs_subset)):
            doc_scores = {}
            if vocab is not None:
                # [Case A] 정답지가 있는 경우: 점수가 0이라도 모든 타겟 단어 기록
                for idx, word in enumerate(feature_names):
                    score = z_matrix[i][idx]
                    doc_scores[word] = round(float(score), 4)
            else:
                # [Case B] 정답지가 없는 경우: 실제 등장한 단어만 기록 (용량 절약)
                nonzero_indices = dense_tfidf[i].nonzero()[0]
                for idx in nonzero_indices:
                    word = feature_names[idx]
                    score = z_matrix[i][idx]
                    doc_scores[word] = round(float(score), 4)
            
            results.append(doc_scores)
            
        return results

    def run(self):
        """
        전체 실행 파이프라인
        1. 데이터 로드 -> 2. Global Z-Score -> 3. Local Z-Score -> 4. 저장
        """
        self.load_all_data()
        if not self.documents: return

        # --- Global Z-Score ---
        self.logger.info("[ZScore] Calculating Global Z-scores...")
        global_scores_full = self._compute_stats(self.documents, vocab=None)
        
        for i, doc in enumerate(self.documents):
            target_vocab = self.domain_target_vocab.get(doc['domain_dir'])
            if target_vocab:
                # 정답지 단어는 점수가 없어도(0.0) 강제로 가져와서 저장
                filtered = {}
                for target_token in target_vocab:
                    score = global_scores_full[i].get(target_token, 0.0)
                    filtered[target_token] = score
                doc['global_z'] = filtered
            else:
                doc['global_z'] = global_scores_full[i]

        # --- Local Z-Score ---
        self.logger.info("[ZScore] Calculating Local Z-scores...")
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