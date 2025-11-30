# src/modules/regex_logics/detectors/card_num_detector.py

import re
from typing import List, Dict, Any
from src.modules.regex_logics.detectors.base import BaseDetector

class CardNumDetector(BaseDetector):
    def __init__(self):
        # 1. 일반 카드번호 패턴 (13~19자리, 하이픈/공백 허용)
        # 기존의 복잡한 BIN(4xxx, 5xxx...) 하드코딩 대신, 
        # "숫자와 하이픈으로 구성된 13~19자리 문자열"을 먼저 찾고 Luhn 알고리즘으로 검증하는 것이
        # 유지보수 면에서 훨씬 강력하고 정확합니다.
        
        # 패턴 설명:
        # (?<!\d): 앞이 숫자가 아니어야 함 (경계)
        # \d{4}: 숫자 4개
        # [- ]?: 하이픈이나 공백이 있거나 없거나
        # ... 반복 ...
        # (?!\d): 뒤가 숫자가 아니어야 함
        self.number_pattern = re.compile(
            r'(?<!\d)(?:\d{4}[- ]?){3}\d{1,4}(?!\d)'  # 13~16자리 일반적인 포맷
        )

        # 2. 마스킹된 카드번호 패턴 (**** 포함)
        # 예: 1234-****-****-1234, 1234 **** **** 1234
        self.masked_pattern = re.compile(
            r'(?<!\d)(?:\d{4}|\*{4})(?:[- ]?(?:\d{4}|\*{4})){3}(?!\d)'
        )

    def detect(self, text: str) -> List[Dict[str, Any]]:
        results = []

        # 1. 일반 카드번호 탐지 + Luhn 알고리즘 검증
        for match in self.number_pattern.finditer(text):
            card_text = match.group()
            
            # 하이픈/공백 제거 후 숫자만 추출
            clean_number = re.sub(r'[^0-9]', '', card_text)
            
            # 길이 체크 (13~19자리) 및 Luhn 알고리즘 통과 여부 확인
            if 13 <= len(clean_number) <= 19 and self._luhn_check(clean_number):
                results.append({
                    "start": match.start(),
                    "end": match.end(),
                    "match": card_text,
                    "label": "카드번호",
                    "score": 1.0  # Luhn 통과면 확실함
                })

        # 2. 마스킹된 카드번호 탐지
        for match in self.masked_pattern.finditer(text):
            card_text = match.group()
            
            # 이미 찾은 일반 카드번호와 겹치는지 확인 (중복 방지)
            is_overlap = any(
                r['start'] == match.start() and r['end'] == match.end() 
                for r in results
            )
            
            if not is_overlap:
                results.append({
                    "start": match.start(),
                    "end": match.end(),
                    "match": card_text,
                    "label": "카드번호",
                    "score": 0.8  # 마스킹 패턴은 꽤 강력하므로 0.5 -> 0.8 상향
                })

        return results

    @staticmethod
    def _luhn_check(card_number: str) -> bool:
        """
        Luhn 알고리즘을 사용한 신용카드 번호 유효성 검사
        (숫자만 들어있는 문자열을 받아야 함)
        """
        # 역순으로 뒤집기
        reverse_digits = card_number[::-1]
        total = 0
        
        for i, digit in enumerate(reverse_digits):
            n = int(digit)
            # 홀수 번째 인덱스 (원래 숫자 기준 짝수 번째 자리)는 2배
            if i % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            total += n
            
        return total % 10 == 0