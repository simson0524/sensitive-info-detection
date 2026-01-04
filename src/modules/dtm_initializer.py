# src/modules/dtm_initializer.py

import os
import json
import pandas as pd
from collections import Counter
from sqlalchemy.orm import Session
from sqlalchemy import text
from transformers import AutoTokenizer
from mecab import MeCab
from tqdm import tqdm # 진행 현황 파악용

from src.database import crud
from src.utils.logger import setup_experiment_logger

class DTMInitializer:
    def __init__(self, session: Session, model_name: str = "klue/roberta-base"):
        """
        [1단계: DTM 초기화 및 기초 데이터 적재]
        - DB를 완전히 비우고, train_data 디렉토리를 스캔하여 도메인과 단어 정보를 구축합니다.
        """
        self.session = session
        self.logger = setup_experiment_logger(experiment_code="DTM_INITIALIZER")
        
        # BERT Tokenizer 및 MeCab 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mecab = MeCab()
        self.special_tokens = set(self.tokenizer.all_special_tokens)
        self.pos_cache = {}
        self.TARGET_TAGS = {'NNG', 'NNP', 'NNB', 'NR', 'SL', 'SN'}

    def _smart_tokenizer(self, text: str):
        """BERT Subword 복원 + MeCab 핵심 품사 필터링 (캐시 적용)"""
        if not text: return []
        raw_tokens = self.tokenizer.tokenize(text)
        merged_chunks = []
        for t in raw_tokens:
            if t in self.special_tokens: continue
            if t.startswith("##"):
                if merged_chunks: merged_chunks[-1] += t[2:]
            else:
                merged_chunks.append(t)
        
        final_tokens = []
        for chunk in merged_chunks:
            if chunk in self.pos_cache:
                final_tokens.extend(self.pos_cache[chunk])
                continue
            try:
                valid_words = [word for word, tag in self.mecab.pos(chunk) if tag in self.TARGET_TAGS]
                self.pos_cache[chunk] = valid_words
                final_tokens.extend(valid_words)
            except: pass
        return final_tokens

    def initialize_and_scan(self, train_data_path: str):
        """
        [데이터 초기화 및 도메인 스캔]
        1. CASCADE 옵션으로 기존 데이터를 모두 TRUNCATE 합니다.
        2. 도메인 폴더를 순회하며 도메인/단어/TF 정보를 DB에 저장합니다.
        """
        self.logger.info("Initializing tables: Truncating domain, term, and DTM...")
        self.session.execute(text("TRUNCATE TABLE domain_term_matrix CASCADE"))
        self.session.execute(text("TRUNCATE TABLE domain CASCADE"))
        self.session.execute(text("TRUNCATE TABLE term CASCADE"))
        self.session.commit()

        abs_train_path = os.path.abspath(train_data_path)
        # {domain_id}_{domain_name} 형태의 디렉토리 목록 확보
        domain_dirs = [d for d in os.listdir(abs_train_path) if os.path.isdir(os.path.join(abs_train_path, d))]

        # [진행바] 도메인 단위 스캔 시작
        for d_dir in tqdm(domain_dirs, desc="[Step 1] Initializing & Scanning Domains", unit="domain"):
            try:
                # 폴더명 파싱 (예: "101_finance")
                parts = d_dir.split('_', 1)
                domain_id = int(parts[0])
                domain_name = parts[1]
                
                # Domain 정보 등록
                crud.create_domain_with_id(self.session, domain_id=domain_id, domain_name=domain_name)

                domain_full_path = os.path.join(abs_train_path, d_dir)
                
                # --- 정답지(Vocabulary) 확보 ---
                target_vocab = set()
                ans_path = os.path.join(domain_full_path, 'answer_sheet.csv')
                if os.path.exists(ans_path):
                    df_ans = pd.read_csv(ans_path)
                    col = df_ans['word'] if 'word' in df_ans.columns else df_ans.iloc[:, 0]
                    for w in col.dropna().astype(str):
                        target_vocab.update(self._smart_tokenizer(w))

                # --- 문서 데이터(JSON) 통합 로드 ---
                all_text = ""
                file_list = [f for f in os.listdir(domain_full_path) if f.endswith('.json')]
                for file_name in file_list:
                    with open(os.path.join(domain_full_path, file_name), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        all_text += " ".join([item.get('sentence', '') for item in data])
                
                # --- 단어 빈도(TF) 계산 ---
                tokens = self._smart_tokenizer(all_text)
                tf_counts = Counter(tokens)

                # 정답지 단어는 빈도가 0이라도 분석 대상에 포함되도록 추가
                for v_word in target_vocab:
                    if v_word not in tf_counts:
                        tf_counts[v_word] = 0

                dtm_inserts = []
                # 도메인 내 개별 단어 처리
                for term, count in tf_counts.items():
                    # Term 마스터 테이블 업데이트 (Included Domain Count 관리)
                    term_info = crud.get_term_stats(self.session, term)
                    if not term_info:
                        crud.bulk_insert_terms(self.session, [{'term': term, 'included_domain_counts': 1}])
                    else:
                        crud.bulk_update_terms(self.session, [{'term': term, 'included_domain_counts': term_info['included_domain_counts'] + 1}])
                    
                    # DTM 기초 레코드 생성
                    dtm_inserts.append({
                        'domain_id': domain_id,
                        'term': term,
                        'tf_score': float(count),
                        'idf_score': 0.0,
                        'tfidf_score': 0.0,
                        'z_score': 0.0
                    })
                
                # 도메인 단위로 DB 벌크 삽입
                crud.bulk_insert_dtm_items(self.session, dtm_inserts)
                
            except Exception as e:
                self.logger.error(f"Error processing {d_dir}: {e}")
                continue