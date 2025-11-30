# src/modules/regex_logics/detectors/birth_age_detector.py

import re
from typing import List, Dict, Any
from src.modules.regex_logics.detectors.base import BaseDetector

class BirthAgeDetector(BaseDetector):
    def __init__(self):
        # (정규식 패턴, 신뢰도 점수) 튜플 리스트
        # 컴파일은 초기화 시 1회만 수행하여 성능 최적화
        self.patterns = [
            # 1. [확실] YYYY년 MM월 DD일 생/출생 (예: 1990년 01월 01일생)
            (re.compile(r"(?:19\d{2}|20\d{2})[.\-/년\s]+(?:0?[1-9]|1[0-2])[.\-/월\s]+(?:0?[1-9]|[12][0-9]|3[01])[일\s]*(?:출생|생)"), 1.0),

            # 2. [확실] YYYY년 생/출생 (예: 1990년생)
            (re.compile(r"(?:19|20)\d{2}(?:년|년도)?\s*(?:생|출생)"), 1.0),

            # 3. [확실] YY년 생/출생 (예: 90년생) - \b로 숫자 앞 경계 확인
            (re.compile(r"\b\d{2}년\s*(?:생|출생)"), 1.0),

            # 4. [보통] 나이 (예: 25세, 30살) - 뒤에 조사가 붙을 수 있으므로 유연하게
            (re.compile(r"\b\d{1,3}(?:세|살)"), 0.5),

            # 5. [애매] YYYY.MM.DD (단순 날짜) - 생일인지 일반 날짜인지 불확실함
            # 문장 끝($) 제약을 풀고, 일반적인 날짜 포맷으로 탐지
            (re.compile(r"(?:19\d{2}|20\d{2})[.\-/년\s]+(?:0?[1-9]|1[0-2])[.\-/월\s]+(?:0?[1-9]|[12][0-9]|3[01])[일\s]*"), 0.4),

            # 6. [매우 애매] YYYY년 (단순 연도)
            (re.compile(r"\b(?:19|20)\d{2}(?:년|년도)?"), 0.2),
        ]

    def detect(self, text: str) -> List[Dict[str, Any]]:
        results = []

        for pattern, base_score in self.patterns:
            for m in pattern.finditer(text):
                results.append({
                    "start": m.start(),
                    "end": m.end(),
                    "match": m.group(),
                    "label": "나이",      # 라벨 통일
                    "score": base_score  # 패턴별 지정된 점수 바로 할당
                })

        return results

    def score(self, match: str) -> float:
        """
        RegexMatcher가 점수를 재확인할 때 사용 (Fallback)
        이미 detect에서 점수를 매겨서 보내므로 호출될 일은 적지만 인터페이스 준수용
        """
        value = match.strip()
        
        # 가장 높은 점수를 주는 패턴부터 검사
        for pattern, base_score in self.patterns:
            # score 함수는 '추출된 문자열' 자체를 검사하므로 fullmatch 사용
            if pattern.fullmatch(value):
                return base_score
                
        return 0.2 # 기본값