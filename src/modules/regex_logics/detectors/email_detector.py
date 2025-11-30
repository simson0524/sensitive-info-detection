# src/modules/regex_logics/detectors/email_detector.py

import re
from typing import List, Dict, Any
from src.modules.regex_logics.detectors.base import BaseDetector

class EmailDetector(BaseDetector):
    def __init__(self):
        # 1. 자주 쓰이는 도메인 확장자 리스트 (TLD)
        # False Positive(오탐)을 줄이기 위해 허용된 TLD만 탐지하도록 제한
        self.allowed_tlds = [
            "com", "net", "org", "co.kr", "ac.kr", "go.kr", "ne.kr", "re.kr",
            "edu", "gov", "mil", "biz", "info", "name", "io",
            "kr", "xyz", "ai", "me", "tech", "site", "online", "store"
        ]

        # 2. TLD 패턴 컴파일
        tld_pattern = '|'.join(map(re.escape, self.allowed_tlds))

        # 3. 이메일 정규식 컴파일
        # 설명:
        # [a-zA-Z0-9_.+*-]+ : 사용자명 (영문, 숫자, 특수문자, 그리고 마스킹용 *)
        # @                : @ 기호
        # [a-zA-Z0-9-]+    : 도메인명 (예: naver, google)
        # (?:\.[a-zA-Z0-9-]+)* : 서브 도메인 (선택적)
        # \.(tld_pattern)  : 허용된 TLD로 끝나야 함 (Case Insensitive flag 사용 예정)
        self.email_pattern = re.compile(
            rf"([a-zA-Z0-9_.+*-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.({tld_pattern}))",
            re.IGNORECASE # 대소문자 구분 없음 (CoM, cOm 등 허용)
        )

    def detect(self, text: str) -> List[Dict[str, Any]]:
        results = []
        
        for match in self.email_pattern.finditer(text):
            email = match.group(1)
            
            # 점수 계산: 마스킹된 이메일은 신뢰도를 약간 낮춤 (문맥에 따라 다를 수 있음)
            # 하지만 이메일 형식이 명확하므로 0.8 정도로 부여
            if "*" in email or "…" in email:
                score = 0.8
            else:
                score = 1.0

            results.append({
                "start": match.start(1),
                "end": match.end(1),
                "match": email,
                "label": "이메일주소",
                "score": score
            })
            
        return results