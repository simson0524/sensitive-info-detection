# src/modules/z_score_calculator.py

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Set, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from transformers import AutoTokenizer
from konlpy.tag import Okt  # [필수] 형태소 분석기 재도입
from src.utils.logger import setup_experiment_logger 

class ZScoreCalculator:
    """
    [Z-Score Calculator]
    BERT Tokenizer의 재결합 능력과 KoNLPy의 형태소 분석 능력을 결합하여
    '조사/어미가 제거된 순수 의미 단어'의 통계적 중요도를 계산합니다.
    """

    def __init__(self, data_root_dir: str = 'data/train_data', model_name: str = "klue/roberta-base"):
        self.logger = setup_experiment_logger(experiment_code="Z_SCORE_CALC")
        self.data_root = data_root_dir
        self.documents: List[Dict[str, Any]] = [] 
        self.domain_target_vocab: Dict[str, Optional[Set[str]]] = {}
        
        # 1. BERT Tokenizer 로드 (서브워드 분해용)
        self.logger.info(f"[ZScore] Loading tokenizer from: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer '{model_name}': {e}")
            raise e
        
        self.special_tokens = set(self.tokenizer.all_special_tokens)
        
        # 2. KoNLPy Okt 로드 (품사 판별 및 조사 제거용)
        self.logger.info("[ZScore] Initializing KoNLPy Okt for POS tagging...")
        self.okt = Okt()

    def _smart_tokenizer(self, text: str) -> List[str]:
        """
        [Advanced Hybrid Tokenizer]
        1. RoBERTa Tokenize: 문장을 모델의 서브워드 단위로 쪼갬
        2. Merge: '##' 토큰을 앞 단어와 결합하여 어절(Eojeol) 단위 복원
        3. POS Check: 복원된 어절을 Okt로 분석하여 조사(Josa) 등을 제거하고 명사/숫자 등만 추출
        
        Example:
            Input: "삼성전자가 2024년에"
            1. RoBERTa: ['삼성', '##전자', '##가', '2024', '##년', '##에']
            2. Merge:   ['삼성전자가', '2024년에']
            3. Okt POS: [('삼성전자', 'Noun'), ('가', 'Josa')], [('2024', 'Number'), ('년', 'Noun'), ('에', 'Josa')]
            4. Filter:  ['삼성전자', '2024', '년']
        """
        if not isinstance(text, str):
            return []
        
        # --- Step 1 & 2: Tokenize & Reconstruct (Merge ##) ---
        raw_tokens = self.tokenizer.tokenize(text)
        merged_chunks = []
        
        for t in raw_tokens:
            if t in self.special_tokens: continue
            
            if t.startswith("##"):
                if merged_chunks:
                    merged_chunks[-1] += t[2:]
                else:
                    merged_chunks.append(t[2:])
            else:
                merged_chunks.append(t)
        
        # --- Step 3 & 4: Morphological Analysis & Filtering ---
        final_tokens = []
        
        # 분석할 의미 있는 품사 정의
        # 1,2,3(명사, 숫자 등)은 살리고 4(조사)는 버리는 기준
        TARGET_TAGS = {'Noun', 'Number', 'Alpha', 'Foreign'} 
        
        for chunk in merged_chunks:
            # 복원된 덩어리(예: "삼성전자가")를 형태소 분석
            # norm=True: 오타 보정, stem=True: 어간 추출
            try:
                pos_results = self.okt.pos(chunk, stem=True, norm=True)
                
                for word, tag in pos_results:
                    if tag in TARGET_TAGS:
                        final_tokens.append(word)
            except Exception:
                # 분석 실패 시(특수문자 등) 원본 청크 그대로 사용 여부 결정 (여기선 스킵)
                pass
                
        return final_tokens

    def load_all_data(self):
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
            
            # --- Answer Sheet CSV ---
            answer_sheet_path = os.path.join(domain_path, 'answer_sheet.csv')
            target_vocab = None
            
            if os.path.exists(answer_sheet_path):
                try:
                    df = pd.read_csv(answer_sheet_path)
                    if 'word' in df.columns: col_data = df['word']
                    else: col_data = df.iloc[:, 0]
                    
                    raw_words = col_data.dropna().astype(str).tolist()
                    processed_vocab = set()
                    
                    for w in raw_words:
                        # 정답지 단어도 동일한 필터링 과정을 거쳐야 매칭됨
                        tokens = self._smart_tokenizer(w)
                        processed_vocab.update(tokens)
                        
                    target_vocab = processed_vocab
                    self.logger.info(f"[ZScore] Domain '{domain_dir}': Loaded {len(target_vocab)} refined words from CSV.")
                    
                except Exception as e:
                    self.logger.warning(f"[ZScore] CSV Error in {domain_dir}: {e}")
            else:
                self.logger.info(f"[ZScore] Domain '{domain_dir}': No CSV. Using auto-extraction mode.")
            
            self.domain_target_vocab[domain_dir] = target_vocab

            # --- Load JSON Docs ---
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
        
        # KoNLPy 처리를 위해 lowercase=False 유지 (영어 대소문자 구분을 위해)
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
        stds[stds == 0] = 1.0 
        
        z_matrix = (dense_tfidf - means) / stds
        
        results = []
        for i in range(len(docs_subset)):
            doc_scores = {}
            if vocab is not None:
                for idx, word in enumerate(feature_names):
                    score = z_matrix[i][idx]
                    doc_scores[word] = round(float(score), 4)
            else:
                nonzero_indices = dense_tfidf[i].nonzero()[0]
                for idx in nonzero_indices:
                    word = feature_names[idx]
                    score = z_matrix[i][idx]
                    doc_scores[word] = round(float(score), 4)
            
            results.append(doc_scores)
            
        return results

    def run(self):
        self.load_all_data()
        if not self.documents: return

        self.logger.info("[ZScore] Calculating Global Z-scores...")
        global_scores_full = self._compute_stats(self.documents, vocab=None)
        
        for i, doc in enumerate(self.documents):
            target_vocab = self.domain_target_vocab.get(doc['domain_dir'])
            if target_vocab:
                filtered = {}
                for target_token in target_vocab:
                    score = global_scores_full[i].get(target_token, 0.0)
                    filtered[target_token] = score
                doc['global_z'] = filtered
            else:
                doc['global_z'] = global_scores_full[i]

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