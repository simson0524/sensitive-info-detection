# src/modules/z_score_calculator.py

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Set, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from src.utils.logger import logger

# [필수] 형태소 분석기 라이브러리 (KoNLPy)
# 이 모듈은 한국어 문장에서 조사, 어미 등을 분리하기 위해 사용됩니다.
from konlpy.tag import Okt 

class ZScoreCalculator:
    """
    [Z-Score Calculator]
    문서 집합에서 TF-IDF를 기반으로 단어의 통계적 중요도(Z-Score)를 계산하는 클래스입니다.
    
    [주요 기능]
    1. Smart Tokenizer: 명사뿐만 아니라 숫자, 영어, 용언 등을 살리고 조사만 제거하는 똑똑한 전처리
    2. Hybrid Approach:
       - Case A (answer_sheet.csv 있음): 정답지에 있는 단어만 타겟팅하여 정밀 분석
       - Case B (answer_sheet.csv 없음): 문서 내 유효한 모든 토큰을 자동 추출하여 분석
    """

    def __init__(self, data_root_dir: str = 'data/train_data'):
        # 데이터가 저장된 최상위 루트 디렉토리
        self.data_root = data_root_dir
        
        # 모든 문서의 원본 데이터와 텍스트를 담을 리스트
        self.documents: List[Dict[str, Any]] = [] 
        
        # 도메인별 타겟 단어장 (Case A: 단어 set, Case B: None)
        self.domain_target_vocab: Dict[str, Optional[Set[str]]] = {}
        
        # [설정] 형태소 분석기 초기화 (속도를 위해 인스턴스는 하나만 생성하여 재사용)
        self.okt = Okt()

    def _smart_tokenizer(self, text: str) -> List[str]:
        """
        [핵심 로직: 스마트 토크나이저]
        단순히 명사만 추출하면 숫자(1, 2024)나 영어(AI) 정보가 사라지는 문제를 해결합니다.
        
        - 역할: 문장을 형태소 단위로 쪼개고, '의미 있는 품사'만 남깁니다.
        - 제거 대상: 조사(Josa), 어미(Eomi), 구두점(Punctuation) 등 분석에 방해되는 노이즈
        - 유지 대상: 명사, 숫자, 알파벳, 외국어, 동사/형용사
        """
        if not isinstance(text, str):
            return []
        
        # 1. 형태소 분석 실행
        # stem=True: '빠른' -> '빠르다', '했다' -> '하다' 처럼 원형으로 복원 (매칭률 상승)
        # norm=True: '임돵' -> '입니다' 처럼 오타/비표준어 정규화
        malist = self.okt.pos(text, stem=True, norm=True)
        
        valid_tokens = []
        
        # 2. 살려둘 품사 정의 (이 목록에 없으면 버려짐)
        # Noun: 명사 ("사과")
        # Number: 숫자 ("1", "2024")
        # Alpha: 영어/알파벳 ("API")
        # Foreign: 외국어
        # Verb/Adjective: 동사/형용사 ("가다", "예쁘다") -> 필요 시 제거 가능
        target_tags = {'Noun', 'Number', 'Alpha', 'Foreign', 'Verb', 'Adjective'}
        
        for word, tag in malist:
            if tag in target_tags:
                valid_tokens.append(word)
                
        return valid_tokens

    def load_all_data(self):
        """
        [데이터 로드 단계]
        1. 각 도메인 폴더를 순회하며 answer_sheet.csv 유무를 확인합니다.
        2. CSV가 있다면(Case A), 단어들을 _smart_tokenizer로 변환하여 저장합니다.
        3. 모든 JSON 문서를 읽어 메모리에 적재합니다.
        """
        abs_root = os.path.abspath(self.data_root)
        logger.info(f"[ZScore] Scanning data root: {abs_root}")
        
        if not os.path.exists(abs_root):
            return

        self.documents = []
        self.domain_target_vocab = {}

        for domain_dir in os.listdir(abs_root):
            domain_path = os.path.join(abs_root, domain_dir)
            
            # 디렉토리가 아니거나 숨김 파일(.git 등)은 패스
            if not os.path.isdir(domain_path) or domain_dir.startswith('.'):
                continue
            
            # --- [Case A/B 분기] answer_sheet.csv 처리 ---
            answer_sheet_path = os.path.join(domain_path, 'answer_sheet.csv')
            target_vocab = None
            
            if os.path.exists(answer_sheet_path):
                # [Case A] 정답지가 존재하는 경우
                try:
                    df = pd.read_csv(answer_sheet_path)
                    # 'word' 컬럼 우선 사용, 없으면 첫 번째 컬럼 사용
                    if 'word' in df.columns:
                        col_data = df['word']
                    else:
                        col_data = df.iloc[:, 0]
                    
                    # [중요] CSV의 단어들도 '본문과 똑같은 기준'으로 토큰화해야 매칭됩니다.
                    # 예: CSV에 "User 1"이라고 써있어도, 토크나이저는 ["User", "1"]로 쪼갭니다.
                    # 따라서 CSV 단어도 쪼개서 저장해야 나중에 본문과 비교(매칭)가 가능합니다.
                    raw_words = col_data.dropna().astype(str).tolist()
                    processed_vocab = set()
                    
                    for w in raw_words:
                        # CSV 단어도 _smart_tokenizer 통과
                        tokens = self._smart_tokenizer(w)
                        processed_vocab.update(tokens)
                        
                    target_vocab = processed_vocab
                    logger.info(f"[ZScore] Domain '{domain_dir}': Loaded {len(target_vocab)} tokens from CSV.")
                    
                except Exception as e:
                    logger.warning(f"[ZScore] CSV Error in {domain_dir}: {e}")
            else:
                # [Case B] 정답지가 없는 경우 -> None으로 설정하여 나중에 '전체 자동 추출' 모드로 동작하게 함
                logger.info(f"[ZScore] Domain '{domain_dir}': No CSV. Using auto-extraction mode.")
            
            self.domain_target_vocab[domain_dir] = target_vocab

            # --- [문서 파일 로드] ---
            # 원본 텍스트를 그대로 저장합니다. 토크나이징은 계산 단계(compute_stats)에서 수행합니다.
            for file_name in os.listdir(domain_path):
                # 결과 파일(z_score.json)이나 json이 아닌 파일은 제외
                if not file_name.endswith('.json') or file_name == 'z_score.json':
                    continue
                
                file_path = os.path.join(domain_path, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 문서 내의 모든 문장을 공백으로 이어붙여 하나의 긴 텍스트로 만듦
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
        입력된 문서들에 대해 TF-IDF 행렬을 만들고 Z-Score를 계산합니다.
        
        Args:
            vocab: 
                - Set[str]: 이 단어들에 대해서만 계산 (Case A)
                - None: 문서에 등장하는 모든 유효 토큰에 대해 계산 (Case B, Global)
        """
        if not docs_subset:
            return []

        corpus = [d['full_text'] for d in docs_subset]
        
        # [핵심] TfidfVectorizer에 커스텀 토크나이저 주입
        # 본문 텍스트가 들어오면 _smart_tokenizer가 조사/어미를 떼고 명사/숫자 등을 리턴합니다.
        vectorizer = TfidfVectorizer(
            tokenizer=self._smart_tokenizer,
            token_pattern=None, # tokenizer 파라미터를 쓸 때는 None이어야 함
            vocabulary=vocab
        )
        
        try:
            # TF-IDF 행렬 생성 (행: 문서, 열: 단어)
            tfidf_matrix = vectorizer.fit_transform(corpus)
        except ValueError:
            # 문서가 비었거나 단어가 하나도 없는 경우 빈 dict 반환
            return [{} for _ in docs_subset]

        feature_names = vectorizer.get_feature_names_out() # 단어 목록
        dense_tfidf = tfidf_matrix.toarray() # 행렬을 numpy array로 변환
        
        # 1. 평균(Mean)과 표준편차(Std) 계산 (Column-wise)
        means = np.mean(dense_tfidf, axis=0)
        stds = np.std(dense_tfidf, axis=0)
        
        # [Zero Division 방지] 표준편차가 0이면(모든 문서에서 값이 같으면) 나눗셈 에러가 나므로 1.0으로 대체
        stds[stds == 0] = 1.0 
        
        # 2. Z-Score 공식: (값 - 평균) / 표준편차
        z_matrix = (dense_tfidf - means) / stds
        
        # 3. 결과 포매팅
        results = []
        for i in range(len(docs_subset)):
            doc_scores = {}
            # 해당 문서에서 값이 0이 아닌(등장한) 단어들의 인덱스만 찾음
            nonzero_indices = dense_tfidf[i].nonzero()[0]
            
            for idx in nonzero_indices:
                word = feature_names[idx]
                score = z_matrix[i][idx]
                doc_scores[word] = round(float(score), 4) # 소수점 4자리까지 저장
            results.append(doc_scores)
            
        return results

    def run(self):
        """
        [실행 파이프라인]
        1. 데이터 로드
        2. Global Z-Score 계산 (전체 데이터 기준) 및 필터링
        3. Local Z-Score 계산 (도메인별 기준)
        4. 결과 저장
        """
        self.load_all_data()
        
        if not self.documents:
            return

        # --- [Step 1] Global Z-Score (전체 문서 대상) ---
        logger.info("[ZScore] Calculating Global Z-scores...")
        
        # vocab=None을 주어, 일단 전체 문서에 등장하는 '모든 유효 토큰'에 대해 통계를 냅니다.
        # 이렇게 해야 전체 말뭉치에서의 희소성(IDF)을 정확히 알 수 있습니다.
        global_scores_full = self._compute_stats(self.documents, vocab=None)
        
        # 계산된 전체 점수에서 '필요한 것'만 남기는 필터링 과정
        for i, doc in enumerate(self.documents):
            target_vocab = self.domain_target_vocab.get(doc['domain_dir'])
            
            if target_vocab:
                # [Case A] 정답지가 있는 도메인:
                # 전체 단어 점수 중, 정답지(CSV)에 있는 단어만 딕셔너리에 남깁니다.
                filtered = {k: v for k, v in global_scores_full[i].items() if k in target_vocab}
                doc['global_z'] = filtered
            else:
                # [Case B] 정답지가 없는 도메인:
                # 토크나이저가 이미 조사/어미를 걸러냈으므로, 남은 모든 명사/숫자 점수를 저장합니다.
                doc['global_z'] = global_scores_full[i]

        # --- [Step 2] Local Z-Score (도메인별 그룹핑) ---
        logger.info("[ZScore] Calculating Local Z-scores...")
        
        # 도메인별로 문서를 묶습니다.
        domain_groups = defaultdict(list)
        doc_indices_map = defaultdict(list)

        for i, doc in enumerate(self.documents):
            domain_groups[doc['domain_dir']].append(doc)
            doc_indices_map[doc['domain_dir']].append(i)

        for domain_dir, group_docs in domain_groups.items():
            target_vocab = self.domain_target_vocab.get(domain_dir)
            
            # [최적화 & 정확도]
            # Case A (target_vocab 존재): 해당 단어들에 대해서만 TF-IDF 행렬을 만듭니다.
            # Case B (target_vocab 없음): 문서 내 자동 추출된 모든 토큰으로 행렬을 만듭니다.
            local_scores = self._compute_stats(group_docs, vocab=target_vocab)
            
            # 계산된 로컬 점수를 원본 문서 객체에 할당
            for local_idx, score_dict in enumerate(local_scores):
                original_idx = doc_indices_map[domain_dir][local_idx]
                self.documents[original_idx]['local_z'] = score_dict

        # --- [Step 3] 결과 저장 ---
        self._save_results()

    def _save_results(self):
        """
        [파일 저장]
        각 도메인 폴더에 'z_score.json' 파일을 생성합니다.
        """
        logger.info("[ZScore] Saving results...")
        
        # 데이터를 도메인별로 구조화
        results_by_domain = defaultdict(dict)
        for doc in self.documents:
            results_by_domain[doc['domain_dir']][doc['doc_id']] = {
                "global": doc.get('global_z', {}),
                "local": doc.get('local_z', {})
            }
        
        # 도메인별로 파일 쓰기
        for domain_dir, data_dict in results_by_domain.items():
            save_path = os.path.join(self.data_root, domain_dir, "z_score.json")
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(data_dict, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Failed to save {save_path}: {e}")