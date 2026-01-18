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
    def __init__(self, session: Session, running_mode: dict, model_name: str = "klue/roberta-base"):
        """
        [Phase 1: 데이터 초기화 및 기초 통계 적재 모듈]
        - DB를 완전히 비우고, 로컬 파일을 스캔하여 도메인/단어/빈도 정보를 구축합니다.
        """
        self.session = session
        self.running_mode = running_mode
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
        
        # --- [Step 1] Running Mode에 따른 DB 초기화 ---
        self.logger.info("Initializing tables: Truncating 'domain_term_matrix'")
        self.session.execute(text("TRUNCATE TABLE domain_term_matrix CASCADE"))

        if self.running_mode['db_truncate']:
            self.logger.info("Initializing tables: Truncating 'domain', 'term'")
            self.session.execute(text("TRUNCATE TABLE domain CASCADE"))
            self.session.execute(text("TRUNCATE TABLE term CASCADE"))
        
        self.session.commit()

        abs_train_path = os.path.abspath(train_data_path)
        # {domain_id}_{domain_name} 구조의 폴더 목록을 가져옵니다.
        domain_dirs = [d for d in os.listdir(abs_train_path) if os.path.isdir(os.path.join(abs_train_path, d))]

        # 전체 단어장을 한 번만 로드 (도메인 루프가 돌아도 재사용)
        full_vocabulary_map = {t['term']: t for t in crud.get_all_terms_streaming(self.session)}    

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
                        target_vocab.add(w.strip())

                # --- [Step 3] 문서 데이터(JSON) 통합 로드 ---      
                """ all_text(str) -> all_documents_text(list[str]) 변경한 점 주의하기!! """
                all_documents_text = []
                # 폴더 내 모든 JSON 파일을 찾아 문장을 all_documents_text에 append 합니다.
                file_list = [f for f in os.listdir(domain_full_path) if f.endswith('.json')]
                for file_name in file_list:
                    if file_name == 'z_score.json': continue # 결과 파일은 제외
                    with open(os.path.join(domain_full_path, file_name), 'r', encoding='utf-8') as f:
                        try:
                            root_data = json.load(f)
                            items = root_data.get('data', []) # {"data": [{"sentence": "..."}]}
                            if isinstance(items, list):
                                all_documents_text.append( " ".join([it.get('sentence', '') for it in items if isinstance(it, dict)]) )
                        except json.JSONDecodeError: continue
                
                # --- [Step 4] 단어 빈도(TF) 계산 ---
                # 본문에서 추출된 모든 단어의 개수를 세어 딕셔너리 형태로 만듭니다.
                tf_counts = Counter()

                for all_text in all_documents_text:
                    tokens = self._smart_tokenizer(all_text)
                    current_counts = Counter(tokens)

                    for v_word in target_vocab:
                        # 1. 본문 텍스트 내에서 해당 단어가 단순히 몇 번 출현하는지 count
                        # (정규표현식이나 단순 count를 사용할 수 있습니다.)
                        appearance_count = all_text.count(v_word)

                        # 2. 만약 형태소 분석 결과(current_counts)보다 직접 센 횟수가 더 많다면 갱신
                        # (이미 current_counts에 있다면 더 큰 값을 취하고, 없다면 새로 등록)
                        if appearance_count > current_counts.get(v_word, 0):
                            current_counts[v_word] = appearance_count

                    # 만약 실행모드가 "presence_count"면, 출현한 문서 개수만 파악해야 하므로 
                    if self.running_mode['mode'] == "presence_count":
                        current_counts = Counter({key: 1 for key in current_counts})

                    tf_counts += current_counts

                # --- [Step 5] domain_term_matrix 및 term 테이블 업데이트 ---
                dtm_inserts = []                
                term_updates = []
                term_inserts = []

                for term, count in tf_counts.items():
                    if term in full_vocabulary_map:
                        # 기존 단어: count만 업데이트 (통계치는 나중에 ZScoreUpdater에서 계산)
                        full_vocabulary_map[term]['included_domain_counts'] += 1
                        term_updates.append({
                            'term': term,
                            'included_domain_counts': full_vocabulary_map[term]['included_domain_counts']
                        })
                    else:
                        # 신규 단어
                        new_term = {
                            'term': term,
                            'included_domain_counts': 1,
                            'avg_tfidf': 0.0, 'stddev_tfidf': 0.0,
                            'sum_tfidf': 0.0, 'sum_square_tfidf': 0.0
                        }
                        # [핵심] 메모리 맵에도 즉시 반영하여 동일 실행 주기 내 중복 INSERT 방지
                        full_vocabulary_map[term] = new_term
                        term_inserts.append(new_term)

                    dtm_inserts.append({
                        'domain_id': domain_id,
                        'term': term,
                        'tf_score': float(count),
                        'idf_score': 0.0, 'tfidf_score': 0.0, 'z_score': 0.0,
                        'is_sensitive_label': term in target_vocab 
                    })
                
                # 도메인 단위로 DB 반영
                if term_inserts: crud.bulk_insert_terms(self.session, term_inserts)
                if term_updates: crud.bulk_update_terms(self.session, term_updates)
                crud.bulk_insert_dtm_items(self.session, dtm_inserts)
                self.session.commit() # 도메인 단위 커밋 권장

            except Exception as e:
                self.logger.error(f"Error in {d_dir}: {e}")
                self.session.rollback()