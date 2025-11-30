# src/modules/regex_logics/detectors/personal_id_detector.py

import re
from typing import List, Dict, Any
from src.modules.regex_logics.detectors.base import BaseDetector

class JuminDetector(BaseDetector):
    def __init__(self):
        # 1. 완전한 주민번호 패턴 (숫자 13자리)
        # 800101-1234567
        self.full_pattern = re.compile(
            r'(?<![0-9a-zA-Z])'           # 앞 경계: 숫자/문자 없어야 함
            r'(\d{2})'                    # YY (연도) - Group 1
            r'(0[1-9]|1[0-2])'            # MM (월) - Group 2
            r'(0[1-9]|[12]\d|3[01])'      # DD (일) - Group 3
            r'-?'                         # 하이픈 (선택적)
            r'([1-8])'                    # 성별 (1~8) - Group 4
            r'(\d{6})'                    # 나머지 6자리 - Group 5
            r'(?![0-9a-zA-Z])'            # 뒤 경계
        )

        # 2. 마스킹된 주민번호 패턴
        # 800101-1******, 800101-*******
        self.masked_pattern = re.compile(
            r'(?<![0-9a-zA-Z])'
            r'(\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])'  # 생년월일
            r'-'
            r'(?:'
            r'[1-8][*xX#]{6}'             # 성별 + 마스킹
            r'|'
            r'[*xX#]{7}'                  # 전체 마스킹
            r')'
            r'(?![0-9a-zA-Z])'
        )

    def detect(self, text: str) -> List[Dict[str, Any]]:
        results = []

        # 1. 전체 주민번호 탐지 (엄격한 검증 수행)
        for match in self.full_pattern.finditer(text):
            full_str = match.group()
            yy, mm, dd, gender, rest = match.groups()
            
            # (1) 날짜 유효성 검사 (윤년 등)
            if not self._is_valid_date(yy, mm, dd, gender):
                continue

            # (2) 체크섬(Checksum) 검사
            # 하이픈 제거 후 숫자만 추출
            clean_nums = full_str.replace('-', '')
            if self._verify_checksum(clean_nums):
                results.append({
                    'start': match.start(),
                    'end': match.end(),
                    'label': '주민번호',
                    'match': full_str,
                    'score': 1.0  # 검증 통과했으므로 확실함
                })

        # 2. 마스킹된 주민번호 탐지 (형식 검사만 수행)
        for match in self.masked_pattern.finditer(text):
            # 이미 찾은 전체 주민번호와 겹치는지 확인 (중복 방지)
            if any(r['start'] == match.start() for r in results):
                continue

            results.append({
                'start': match.start(),
                'end': match.end(),
                'label': '주민번호',
                'match': match.group(),
                'score': 0.8  # 마스킹은 체크섬 불가하므로 점수 낮춤
            })

        return results

    def _is_valid_date(self, yy: str, mm: str, dd: str, gender: str) -> bool:
        """
        생년월일이 달력상 유효한지 검사합니다. (윤년 체크 포함)
        """
        year = int(yy)
        month = int(mm)
        day = int(dd)
        g = int(gender)

        # 1900년대: 1, 2, 5, 6 / 2000년대: 3, 4, 7, 8
        if g in [1, 2, 5, 6]:
            full_year = 1900 + year
        elif g in [3, 4, 7, 8]:
            full_year = 2000 + year
        else:
            full_year = 1900 + year # 기본 처리

        try:
            # 월별 일수 체크
            if month < 1 or month > 12:
                return False
                
            days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            
            # 윤년 체크 (4년마다 O, 100년마다 X, 400년마다 O)
            if (full_year % 4 == 0 and full_year % 100 != 0) or (full_year % 400 == 0):
                days_in_month[2] = 29

            return 1 <= day <= days_in_month[month]
        except:
            return False

    def _verify_checksum(self, nums: str) -> bool:
        """
        주민등록번호 마지막 자리 검증 로직
        """
        if len(nums) != 13:
            return False

        # 공식: (2*A + 3*B + ... + 5*L) % 11
        # 11 - 결과 = 마지막 자리
        multipliers = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]
        
        try:
            total = sum(int(nums[i]) * multipliers[i] for i in range(12))
            remainder = total % 11
            check_digit = (11 - remainder) % 10
            
            return check_digit == int(nums[12])
        except ValueError:
            return False