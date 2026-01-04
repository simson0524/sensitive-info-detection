# src/modules/dtm_initializer.py

import os
import json
import pandas as pd
from collections import Counter
from sqlalchemy.orm import Session
from sqlalchemy import text
from transformers import AutoTokenizer
from mecab import MeCab
from tqdm import tqdm

from src.database import crud
from src.utils.logger import setup_experiment_logger

class DTMInitializer:
    def __init__(self, session: Session, model_name: str = "klue/roberta-base"):
        """
        [Phase 1: 데이터 초기화 및 기초 통계 적재 모듈]
        - DB를 완전히 비우고, 로컬 파일을 스캔하여 도메인/단어/빈도 정보를 구축합니다.
        """
        self.session = session
        self.logger = setup_experiment_logger(experiment_code="DTM_INITIALIZER")
        
        # 1. BERT Tokenizer 로드 (Subword 분해 및 재결합용)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 2. MeCab 초기화 (형태소 분석 및 핵심 품사 필터링용)
        self.mecab = MeCab()
        # 3. 특수 토큰 세트 (분석에서 제외할 토큰들)
        self.special_tokens = set(self.tokenizer.all_special_tokens)
        
        # [최적화] 형태소 분석 결과 캐시 (메모이제이션으로 속도 향상)
        self.pos_cache = {}
        # [필터링] 추출할 핵심 품사 태그 (명사, 숫자, 외국어)
        self.TARGET_TAGS = {'NNG', 'NNP', 'NNB', 'NR', 'SL', 'SN'}

    def _smart_tokenizer(self, text: str):
        """
        [하이브리드 토큰화 로직]
        - BERT 토크나이저로 쪼갠 후, 서브워드(##)를 앞 단어와 합쳐 어절로 복원합니다.
        - 복원된 어절을 MeCab으로 분석하여 조사/어미를 떼어내고 '알맹이' 단어만 남깁니다.
        """
        if not text: return []
        
        # Step 1: BERT Subword Merge (예: ['삼성', '##전자'] -> ['삼성전자'])
        raw_tokens = self.tokenizer.tokenize(text)
        merged_chunks = []
        for t in raw_tokens:
            if t in self.special_tokens: continue
            if t.startswith("##"):
                if merged_chunks: merged_chunks[-1] += t[2:]
            else:
                merged_chunks.append(t)
        
        # Step 2: MeCab POS Filtering (의미 있는 품사만 추출)
        final_tokens = []
        for chunk in merged_chunks:
            # 캐시 확인: 이미 분석한 단어라면 캐시값 사용
            if chunk in self.pos_cache:
                final_tokens.extend(self.pos_cache[chunk])
                continue
            
            try:
                # 형태소 분석 수행 후 지정된 태그(TARGET_TAGS)만 리스트에 담음
                valid_words = [word for word, tag in self.mecab.pos(chunk) if tag in self.TARGET_TAGS]
                self.pos_cache[chunk] = valid_words
                final_tokens.extend(valid_words)
            except: pass
        return final_tokens

    def initialize_and_scan(self, train_data_path: str):
        """
        [전체 실행 파이프라인]
        1. 기존 테이블 데이터 초기화 (CASCADE TRUNCATE)
        2. 도메인 폴더별 스캔 및 도메인 정보 등록
        3. 정답지(CSV)를 통한 민감 단어 목록(target_vocab) 확보
        4. 문서(JSON) 텍스트 추출 및 단어 빈도(TF) 산출
        5. 정답지 포함 여부에 따른 is_sensitive_label 설정 및 DB 적재
        """
        
        # --- [Step 1] DB 완전 초기화 ---
        # CASCADE를 사용하여 외래 키 관계에 있는 테이블들을 모두 깨끗이 비웁니다.
        self.logger.info("Initializing tables: Truncating domain, term, and DTM...")
        self.session.execute(text("TRUNCATE TABLE domain_term_matrix CASCADE"))
        self.session.execute(text("TRUNCATE TABLE domain CASCADE"))
        self.session.execute(text("TRUNCATE TABLE term CASCADE"))
        self.session.commit()

        abs_train_path = os.path.abspath(train_data_path)
        # {domain_id}_{domain_name} 구조의 폴더 목록을 가져옵니다.
        domain_dirs = [d for d in os.listdir(abs_train_path) if os.path.isdir(os.path.join(abs_train_path, d))]

        # [진행바] 도메인 단위로 스캔을 시작합니다.
        for d_dir in tqdm(domain_dirs, desc="[Step 1] Initializing & Scanning Domains", unit="domain"):
            try:
                # 폴더명을 분리하여 ID와 이름을 추출합니다.
                parts = d_dir.split('_', 1)
                domain_id = int(parts[0])
                domain_name = parts[1]
                
                # 1. Domain 테이블 레코드 생성
                crud.create_domain_with_id(self.session, domain_id=domain_id, domain_name=domain_name)
                domain_full_path = os.path.join(abs_train_path, d_dir)
                
                # --- [Step 2] 정답지(answer_sheet.csv) 어휘 확보 ---
                # 이 어휘들은 문서에 등장하지 않더라도 DTM에 등록되며, is_sensitive_label이 True가 됩니다.
                target_vocab = set()
                ans_path = os.path.join(domain_full_path, 'answer_sheet.csv')
                if os.path.exists(ans_path):
                    df_ans = pd.read_csv(ans_path)
                    col = df_ans['word'] if 'word' in df_ans.columns else df_ans.iloc[:, 0]
                    for w in col.dropna().astype(str):
                        # 정답지의 단어도 본문 분석과 동일한 토크나이저를 거쳐 일관성을 유지합니다.
                        target_vocab.update(self._smart_tokenizer(w))

                # --- [Step 3] 문서 데이터(JSON) 통합 로드 ---
                all_text = ""
                # 폴더 내 모든 JSON 파일을 찾아 문장을 합칩니다.
                file_list = [f for f in os.listdir(domain_full_path) if f.endswith('.json')]
                for file_name in file_list:
                    if file_name == 'z_score.json': continue # 결과 파일은 제외
                    with open(os.path.join(domain_full_path, file_name), 'r', encoding='utf-8') as f:
                        try:
                            root_data = json.load(f)
                            items = root_data.get('data', []) # {"data": [{"sentence": "..."}]}
                            if isinstance(items, list):
                                all_text += " " + " ".join([it.get('sentence', '') for it in items if isinstance(it, dict)])
                        except json.JSONDecodeError: continue
                
                # --- [Step 4] 단어 빈도(TF) 계산 ---
                # 본문에서 추출된 모든 단어의 개수를 세어 딕셔너리 형태로 만듭니다.
                tokens = self._smart_tokenizer(all_text)
                tf_counts = Counter(tokens)

                # [중요] 정답지 단어 보정: 본문에는 없지만 정답지에는 있는 단어들을 TF 0으로 추가합니다.
                # 이를 통해 정답지 단어들이 통계 분석 대상(DTM)에 반드시 포함되도록 합니다.
                for v_word in target_vocab:
                    if v_word not in tf_counts:
                        tf_counts[v_word] = 0

                dtm_inserts = []
                # --- [Step 5] DTM(7번) 및 Term(9번) 마스터 업데이트 ---
                for term, count in tf_counts.items():
                    # 9번 테이블(Term) 업데이트: 해당 단어가 시스템 전체에서 몇 개의 도메인에 등장했는지 기록
                    term_info = crud.get_term_stats(self.session, term)
                    if not term_info:
                        # 처음 발견된 단어면 새로 등록
                        crud.bulk_insert_terms(self.session, [{'term': term, 'included_domain_counts': 1}])
                    else:
                        # 이미 있는 단어면 도메인 카운트만 +1
                        crud.bulk_update_terms(self.session, [{'term': term, 'included_domain_counts': term_info['included_domain_counts'] + 1}])
                    
                    # 7번 테이블(DTM) 삽입용 데이터 리스트 구성
                    dtm_inserts.append({
                        'domain_id': domain_id,
                        'term': term,
                        'tf_score': float(count),
                        'idf_score': 0.0,
                        'tfidf_score': 0.0,
                        'z_score': 0.0,
                        # [핵심 로직] 현재 단어가 정답지(target_vocab)에 있으면 True, 아니면 False
                        'is_sensitive_label': term in target_vocab 
                    })
                
                # 도메인 단위로 데이터를 DB에 일괄 삽입(Bulk Insert)합니다.
                crud.bulk_insert_dtm_items(self.session, dtm_inserts)
                
            except Exception as e:
                # 오류 발생 시 로그를 남기고 다음 도메인으로 넘어갑니다.
                self.logger.error(f"Error processing {d_dir}: {e}")
                continue