# src/modules/regex_logics/detectors/phone_num_detector.py

import re
from typing import List, Dict, Any
from src.modules.regex_logics.detectors.base import BaseDetector

class PhoneDetector(BaseDetector):
    def __init__(self):
        # 1. 휴대폰 번호 패턴
        # 010, 011, 016, 017, 018, 019 허용
        # 중간/끝자리 마스킹(*, X) 허용
        self.mobile_pattern = re.compile(
            r'\b(01[016789])'                 # 식별번호 (010~019)
            r'[-\s]?'                         # 구분자 (하이픈, 공백, 없음)
            r'(\d{3,4}|[*xX]{3,4})'           # 국번 (3~4자리 숫자 or 마스킹)
            r'[-\s]?'                         # 구분자
            r'(\d{4}|[*xX]{4})\b'             # 번호 (4자리 숫자 or 마스킹)
        )

        # 2. 유선전화 및 기타 번호 패턴
        # 지역번호 + 인터넷전화(070) + 안심번호(050)
        self.area_codes = [
            '02', '031', '032', '033', '041', '042', '043', '044',
            '051', '052', '053', '054', '055', '061', '062', '063', '064',
            '070', '050' # 070, 050 추가
        ]
        
        # 리스트를 정규식 OR 패턴으로 변환 (긴 것부터 매칭되도록 정렬할 필요는 없으나 명시적임)
        area_code_pattern = '|'.join(self.area_codes)

        self.landline_pattern = re.compile(
            rf'\b({area_code_pattern})'       # 지역번호/식별번호
            rf'[-\s]?'                        # 구분자
            rf'(\d{3,4})'                     # 국번
            rf'[-\s]?'                        # 구분자
            rf'(\d{4})\b'                     # 번호
        )

    def detect(self, text: str) -> List[Dict[str, Any]]:
        results = []

        # 1. 휴대폰 번호 탐지
        for match in self.mobile_pattern.finditer(text):
            results.append({
                'start': match.start(),
                'end': match.end(),
                'label': '전화번호',
                'match': match.group(), # 원본 문자열 그대로 사용 (예: "010 1234 5678")
                'score': 1.0            # 휴대폰 번호 형식은 식별성이 매우 높음
            })

        # 2. 유선전화/인터넷전화 탐지
        for match in self.landline_pattern.finditer(text):
            # 이미 찾은 휴대폰 번호 영역과 겹치는지 확인 (혹시 모를 오탐 방지)
            is_overlap = any(
                r['start'] == match.start() and r['end'] == match.end()
                for r in results
            )
            
            if not is_overlap:
                results.append({
                    'start': match.start(),
                    'end': match.end(),
                    'label': '전화번호',
                    'match': match.group(),
                    'score': 0.8        # 지역번호는 우연히 겹칠 확률이 조금 더 있음
                })

        return results