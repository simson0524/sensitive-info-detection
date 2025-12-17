# src/modules/confidence_score_calculator.py

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from konlpy.tag import Okt
from typing import Tuple, Set, Dict, Any

class ConfidenceScoreCalculator:
    """
    Log-Odds Ratio with Dirichlet Prior 알고리즘을 사용하여
    특정 도메인(Inner)과 타 도메인(Outer) 간의 단어 중요도(Confidence Score)를 계산하는 모듈.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        :param config: 설정 딕셔너리 (alpha, target_domain_id 등 포함)
        """
        self.config = config
        self.alpha = config.get("alpha", 10.0)  # 스무딩 파라미터 (기본값 10)
        self._tokenizer = None  # Lazy Initialization

    @property
    def tokenizer(self):
        """Okt 인스턴스를 필요할 때만 생성 (메모리/속도 효율성)"""
        if self._tokenizer is None:
            print("Initializing Okt tokenizer...")
            self._tokenizer = Okt()
        return self._tokenizer

    def _extract_nouns(self, file_path: Path) -> list:
        """JSON 파일에서 'data' -> 'sentence'를 추출하여 명사 리스트 반환"""
        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                # 데이터 구조에 맞춰 파싱
                sentence = data.get("data", {}).get("sentence", "")
                if not sentence:
                    return []
                return self.tokenizer.nouns(sentence)
        except Exception as e:
            print(f"Failed to read/parse {file_path}: {e}")
            return []

    def _scan_and_count(self, data_root: Path, target_domain_id: str, target_words: Set[str]) -> Tuple[int, int, Counter, Counter]:
        """
        데이터 폴더 전체를 1회 스캔하여 Inner/Outer 통계를 집계 (Single Pass)
        
        :param data_root: data/train_data/ 폴더 경로
        :param target_domain_id: 분석 대상(Inner) 도메인 ID (예: '08')
        :param target_words: 카운팅할 대상 단어 집합 (csv에 있는 단어들)
        """
        n_in, n_out = 0, 0
        in_counts = Counter()
        out_counts = Counter()

        # data/train_data/ 하위의 모든 json 검색
        # 구조: {domain_id}_{domain_name}/{domain_id}_{doc_id}.json
        files = list(data_root.rglob("*.json"))
        total_files = len(files)
        
        print(f"Start scanning {total_files} files in {data_root} for Domain ID: {target_domain_id}")

        for idx, fp in enumerate(files):
            if idx > 0 and idx % 1000 == 0:
                print(f"Processed {idx}/{total_files} files...")

            # 파일명에서 도메인 ID 추출 (파일명 규칙: {domain_id}_{doc_id}.json)
            # 예: 08_contract_001.json -> 08
            parts = fp.stem.split("_")
            if not parts:
                continue
            
            # 첫 번째 파트가 domain_id라고 가정
            current_domain_id = parts[0]

            # 1. 명사 토큰 추출
            tokens = self._extract_nouns(fp)
            if not tokens:
                continue
            
            # 2. 통계 집계
            # 전체 토큰 수 (N) 계산 (타겟 단어 여부와 상관없이)
            token_count = len(tokens)

            # 타겟 단어 빈도 계산 (필요한 단어만 필터링하여 카운트 - 속도 최적화)
            current_counts = Counter([t for t in tokens if t in target_words])

            if current_domain_id == target_domain_id:
                # Inner Domain (내부)
                n_in += token_count
                in_counts.update(current_counts)
            else:
                # Outer Domain (외부/배경)
                n_out += token_count
                out_counts.update(current_counts)

        print(f"Scan Completed. [Inner(ID:{target_domain_id}) Total Tokens: {n_in}] | [Outer Total Tokens: {n_out}]")
        return n_in, n_out, in_counts, out_counts

    def calculate(self, data_root: Path, target_domain_id: str, word_list_path: Path, output_path: Path) -> pd.DataFrame:
        """
        전체 파이프라인 실행: CSV 로드 -> 통계 집계 -> Z-score 계산 -> CSV 저장
        
        :param data_root: 학습 데이터 루트 폴더 (data/train_data)
        :param target_domain_id: 타겟 도메인 ID (예: '08')
        :param word_list_path: 정답지/단어목록 CSV 경로
        :param output_path: 결과 저장 경로 (confidence_score.csv)
        """
        # 1. 대상 단어 목록 로드
        if not word_list_path.exists():
            raise FileNotFoundError(f"Word list not found: {word_list_path}")

        print(f"Loading word list from {word_list_path}")
        df_words = pd.read_csv(word_list_path)
        # 컬럼명 공백 제거
        df_words.columns = df_words.columns.str.strip()
        
        # '단어' 컬럼 확인
        if "단어" not in df_words.columns:
             raise ValueError("CSV file must contain a column named '단어'")

        # 빠른 조회를 위해 set으로 변환
        target_words_set = set(df_words["단어"].tolist())

        # 2. 통계 집계 (병목 지점 - 최적화 적용됨)
        n_in, n_out, in_counts, out_counts = self._scan_and_count(data_root, target_domain_id, target_words_set)

        if n_in == 0:
            print("No tokens found for the target domain. Check domain_id or data path.")
            return pd.DataFrame()

        # 3. 점수 계산 (Vectorized Operations)
        print("Computing Confidence Scores...")
        
        # Counter를 DataFrame 컬럼으로 매핑 (O(1) Lookup)
        df_words['x_in'] = df_words['단어'].map(in_counts).fillna(0)
        df_words['x_out'] = df_words['단어'].map(out_counts).fillna(0)

        EPS = 1e-8
        alpha = self.alpha

        # 배경 확률 (Background Probability)
        # Outer 데이터가 없으면 Uniform distribution 가정 방어 로직 추가 가능
        if n_out == 0:
            print("No outer domain data found. Using uniform distribution for background.")
            n_out = 1
            mu_w = EPS
        else:
            mu_w = df_words['x_out'] / n_out
            mu_w = np.maximum(mu_w, EPS)

        # Smoothed Probabilities (Dirichlet Prior)
        p_in = (df_words['x_in'] + alpha * mu_w) / (n_in + alpha)
        p_out = (df_words['x_out'] + alpha * mu_w) / (n_out + alpha)

        # Clipping (안전 범위)
        p_in = np.clip(p_in, EPS, 1.0 - EPS)
        p_out = np.clip(p_out, EPS, 1.0 - EPS)

        # Log-Odds Ratio
        log_odds_in = np.log(p_in / (1.0 - p_in))
        log_odds_out = np.log(p_out / (1.0 - p_out))
        delta = log_odds_in - log_odds_out

        # Variance Approximation (Monroe et al. 2008)
        var_delta = (
            (1.0 / (df_words['x_in'] + alpha * mu_w)) + 
            (1.0 / (n_in - df_words['x_in'] + alpha * (1 - mu_w))) +
            (1.0 / (df_words['x_out'] + alpha * mu_w)) + 
            (1.0 / (n_out - df_words['x_out'] + alpha * (1 - mu_w)))
        )

        # Z-Score (Confidence Score)
        df_words['conf_score'] = delta / np.sqrt(var_delta)

        # 부가 정보 저장 (디버깅용)
        df_words['p_in'] = p_in
        df_words['p_out'] = p_out
        
        # 4. 결과 저장
        # 상위 폴더가 없으면 생성
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df_words.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Confidence Score calculation completed. Saved to: {output_path}")
        
        return df_words