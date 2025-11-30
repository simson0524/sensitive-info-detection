# src/modules/regex_logics/detectors/address_detector.py

import re
from typing import List, Dict, Any, Set
from src.modules.regex_logics.detectors.base import BaseDetector

class AddressDetector(BaseDetector):
    """
    주소(Address) 정보를 탐지하는 클래스
    시도, 시군구, 읍면동, 도로명, 건물번호 등을 복합적으로 분석합니다.
    """

    def __init__(self, sido_list: Set[str], sigungu_list: Set[str], dong_list: Set[str]):
        """
        Args:
            sido_list (Set[str]): 시/도 목록 (Set)
            sigungu_list (Set[str]): 시/군/구 목록 (Set)
            dong_list (Set[str]): 읍/면/동 목록 (Set)
        """
        # 1. 리스트 기반 패턴 컴파일 (특수문자 이스케이프 처리)
        # Set은 순서가 없으므로, 긴 단어가 먼저 매칭되도록 정렬(Sort)하여 리스트로 변환 후 컴파일
        # 예: '강원'보다 '강원특별자치도'가 먼저 매칭되어야 함
        
        sorted_sido = sorted(list(sido_list), key=len, reverse=True) if sido_list else []
        sorted_sigungu = sorted(list(sigungu_list), key=len, reverse=True) if sigungu_list else []
        sorted_dong = sorted(list(dong_list), key=len, reverse=True) if dong_list else []

        sido_pat = "|".join(map(re.escape, sorted_sido)) if sorted_sido else "impossible_placeholder"
        sigungu_pat = "|".join(map(re.escape, sorted_sigungu)) if sorted_sigungu else "impossible_placeholder"
        dong_pat = "|".join(map(re.escape, sorted_dong)) if sorted_dong else "impossible_placeholder"

        self.sido_pattern = re.compile(sido_pat)
        self.sigungu_pattern = re.compile(sigungu_pat)
        self.dong_pattern = re.compile(dong_pat)

        # 2. 규칙 기반 패턴 컴파일
        self.road_pattern = re.compile(r"[가-힣0-9]+(?:로|길|대로)")
        self.apt_pattern = re.compile(r"(?:[가-힣0-9]+)?(?:아파트|빌딩|타워|오피스텔|기숙사|본관|별관)")
        self.dong_num_pattern = re.compile(r"\d+동")
        self.ho_num_pattern = re.compile(r"\d+호")
        self.addr_num_pattern = re.compile(r"\d+(?:-\d+)?(?:번지)?")

        # 3. 통합 정규식 구성 (Named Groups 사용)
        # (?P<이름>패턴) 형식을 사용하여 그룹 인덱스(1,2..) 대신 이름으로 접근
        self.address_block_pattern = re.compile(
            rf"(?P<sido>{self.sido_pattern.pattern})?\s*"              # 시도
            rf"(?P<sigungu1>{self.sigungu_pattern.pattern})?\s*"       # 시군구1
            rf"(?P<sigungu2>{self.sigungu_pattern.pattern})?\s*"       # 시군구2
            rf"(?P<dong>{self.dong_pattern.pattern})?\s*"              # 동/읍/면
            rf"(?P<road>{self.road_pattern.pattern})?\s*"              # 도로명
            rf"(?P<addr_num>{self.addr_num_pattern.pattern})?\s*"      # 번지
            rf"(?P<building>{self.apt_pattern.pattern})?\s*"           # 건물명
            rf"(?P<dong_num>{self.dong_num_pattern.pattern})?\s*"      # 동번호
            rf"(?P<ho_num>{self.ho_num_pattern.pattern})?"             # 호수
        )

        # 4. 라벨 매핑 (그룹명 -> 출력 라벨)
        self.group_to_label = {
            "sido": "도시",
            "sigungu1": "도, 주",
            "sigungu2": "도, 주",
            "dong": "군, 면, 동",
            "road": "도로명",
            "addr_num": "주소숫자",
            "building": "건물명",
            "dong_num": "주소숫자",
            "ho_num": "주소숫자"
        }

    def _calculate_score(self, match_count: int) -> float:
        """매칭된 컴포넌트 개수에 따라 신뢰도 점수 계산 (내부 로직)"""
        if match_count >= 5:
            return 1.0
        elif match_count == 4:
            return 0.8
        elif 2 <= match_count <= 3:
            return 0.5
        else:
            return 0.2

    # [참고] BaseDetector의 추상 메서드 구현
    def detect(self, text: str) -> List[Dict[str, Any]]:
        results = []

        for m in self.address_block_pattern.finditer(text):
            # 매칭된 그룹 필터링
            captured_groups = {k: v for k, v in m.groupdict().items() if v}

            if not captured_groups:
                continue

            # --- 유효성 검사 ---
            has_sigungu = 'sigungu1' in captured_groups or 'sigungu2' in captured_groups
            has_dong = 'dong' in captured_groups
            has_road = 'road' in captured_groups
            has_addrnum = 'addr_num' in captured_groups

            # 1. 도로명 유효성 체크
            if has_road and not (has_dong or has_sigungu):
                continue

            # 2. 번지수 유효성 체크
            if has_addrnum and not (has_road or has_dong):
                continue

            # --- 점수 계산 ---
            score = self._calculate_score(len(captured_groups))

            for group_name, match_text in captured_groups.items():
                label = self.group_to_label.get(group_name, "기타")
                
                results.append({
                    "start": m.start(group_name),
                    "end": m.end(group_name),
                    "match": match_text,
                    "label": label,
                    "score": score
                })

        return results