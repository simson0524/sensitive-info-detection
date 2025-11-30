# src/modules/regex_logics/detectors/name_detector.py

import re
from typing import List, Dict, Any, Set
from src.modules.regex_logics.detectors.base import BaseDetector

class NameDetector(BaseDetector):
    def __init__(
        self, 
        surnames: Set[str], 
        first_names: Set[str], 
        last_names: Set[str], 
        single_names: Set[str], 
        stopwords: Set[str] = None
    ):
        """
        Args:
            surnames: 성(Surname) 집합 (Set)
            first_names: 이름 첫 글자 집합 (Set)
            last_names: 이름 끝 글자 집합 (Set)
            single_names: 외자 이름 집합 (Set)
            stopwords: 불용어 집합 (Set)
        """
        # 입력받은 데이터가 이미 Set이므로 그대로 할당 (불필요한 형변환 제거)
        self.surnames = surnames
        self.first_names = first_names
        self.last_names = last_names
        self.single_names = single_names
        self.stopwords = stopwords if stopwords else set()

        # 조사 리스트 (제거용)
        self.postpositions = {'은', '는', '이', '가', '도', '의', '에게', '께'}

        # 1. 정규식 구성
        # 성(Surname)을 앵커로 잡고 뒤에 오는 글자 패턴을 찾음
        # Set은 순서가 없으므로 정규식 컴파일을 위해 리스트로 변환 후 정렬(sort)하는 것이 안전함
        # (re.escape를 적용하여 특수문자 오류 방지)
        sorted_surnames = sorted(list(self.surnames), key=len, reverse=True)
        sn_pattern = '|'.join(map(re.escape, sorted_surnames))
        
        # 이름 부분: 한글, 영문, 마스킹기호 등 1~3글자
        name_char_class = r"[가-힣a-zA-Z*○△□0-9]"
        
        self.general_pattern = re.compile(
            rf"(?P<surname>{sn_pattern})(?P<firstname>{name_char_class}{{1,3}})"
        )

        # 마스킹 패턴 정의 (미리 컴파일)
        self.mask_patterns = [
            (re.compile(r'[가-힣][0○△□]{2}'), 0.5),      # 김00
            (re.compile(r'[가-힣]모씨?'), 0.5),           # 김모, 김모씨
            (re.compile(r'[가-힣]\s모씨?'), 0.5),          # 김 모씨
            (re.compile(r'[가-힣][Xx*#]{2}'), 0.5),       # 김**
        ]

    def _is_valid_name_combination(self, surname: str, firstname: str) -> float:
        """
        성+이름 조합이 유효한지 검사하고 점수를 반환합니다.
        """
        full_name = surname + firstname
        name_len = len(full_name)

        # 1. 길이 체크
        if name_len < 2 or name_len > 4:
            return 0.0
        
        # 2. 불용어 체크 (O(1) 속도)
        if full_name in self.stopwords:
            return 0.0

        # 3. 3글자 이름 (성 + 중간 + 끝)
        if name_len == 3:
            # 3-1. 완전 일치 (김철수)
            if firstname[0] in self.first_names and firstname[1] in self.last_names:
                return 1.0
            
            # 3-2. 마스킹 처리 (김*수)
            if firstname[0] == '*' and firstname[1] in self.last_names:
                return 0.8
            # 3-3. 마스킹 처리 (김철*)
            if firstname[0] in self.first_names and firstname[1] == '*':
                return 0.8
            # 3-4. 마스킹 처리 (김**)
            if firstname == '**':
                return 0.5
            
            # 3-5. 기타 마스킹 패턴 확인
            for pat, score in self.mask_patterns:
                if pat.fullmatch(full_name):
                    return score

        # 4. 2글자 이름 (성 + 외자)
        if name_len == 2:
            if firstname in self.single_names:
                return 1.0
            
        return 0.0

    def detect(self, text: str) -> List[Dict[str, Any]]:
        results = []
        
        for m in self.general_pattern.finditer(text):
            surname = m.group('surname')
            firstname_raw = m.group('firstname')
            
            # 조사 제거 로직
            firstname = firstname_raw
            if len(firstname) > 1 and firstname[-1] in self.postpositions:
                firstname = firstname[:-1]

            full_match = surname + firstname
            
            score = self._is_valid_name_combination(surname, firstname)
            
            if score > 0.0:
                results.append({
                    "start": m.start(),
                    "end": m.start() + len(full_match),
                    "match": full_match,
                    "label": "인물",
                    "score": score
                })

        return results