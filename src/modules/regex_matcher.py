# src/modules/regex_matcher.py

from typing import List, Dict, Any

# 1. Detector Modules Import
from src.modules.regex_logics.detectors.address_detector import AddressDetector
from src.modules.regex_logics.detectors.birth_age_detector import BirthAgeDetector
from src.modules.regex_logics.detectors.email_detector import EmailDetector
from src.modules.regex_logics.detectors.personal_id_detector import JuminDetector
from src.modules.regex_logics.detectors.phone_num_detector import PhoneDetector
from src.modules.regex_logics.detectors.card_num_detector import CardNumDetector
from src.modules.regex_logics.detectors.name_detector import NameDetector

# 2. Dictionary Data Import (대문자 상수로 변경됨)
from src.modules.regex_logics.dictionary.address_dict import SIDO_LIST, SIGUNGU_LIST, DONG_LIST
from src.modules.regex_logics.dictionary.name_dict import SURNAMES, FIRST_NAMES, LAST_NAMES, SINGLE_NAMES
from src.modules.regex_logics.dictionary.stopwords_dict import STOPWORDS

class RegexMatcher:
    """
    정규표현식 및 규칙 기반(Rule-based) PII 탐지 모듈
    여러 개의 Detector를 통합 관리하고 실행합니다.
    """

    # 탐지된 라벨에 대한 분류 매핑 (개인/기밀, 식별/준식별)
    DETECTOR_TYPE_MAP = {
        "인물": {"category": "개인", "type": "식별"},
        "도시": {"category": "개인", "type": "준식별"},
        "카드번호": {"category": "개인", "type": "준식별"},
        "도, 주": {"category": "개인", "type": "준식별"},
        "군, 면, 동": {"category": "개인", "type": "준식별"},
        "도로명": {"category": "개인", "type": "준식별"},
        "건물명": {"category": "개인", "type": "준식별"},
        "주소숫자": {"category": "개인", "type": "준식별"},
        "나이": {"category": "개인", "type": "식별"},
        "이메일주소": {"category": "개인", "type": "식별"},
        "주민번호": {"category": "개인", "type": "식별"},
        "전화번호": {"category": "개인", "type": "식별"},
    }

    def __init__(self):
        """
        Detector들을 초기화합니다.
        사전 데이터(Dictionary)를 각 Detector에 주입합니다.
        """
        print("🛠 [RegexMatcher] Initializing detectors...")
        
        self.detectors = [
            # 1. 주소 탐지기 (Set 데이터 주입)
            AddressDetector(
                sido_list=SIDO_LIST,
                sigungu_list=SIGUNGU_LIST,
                dong_list=DONG_LIST
            ),
            
            # 2. 인물 탐지기 (Set 데이터 주입)
            NameDetector(
                surnames=SURNAMES,
                first_names=FIRST_NAMES,
                last_names=LAST_NAMES,
                single_names=SINGLE_NAMES,
                stopwords=STOPWORDS
            ),

            # 3. 기타 규칙 기반 탐지기들 (데이터 주입 불필요)
            BirthAgeDetector(),
            EmailDetector(),
            JuminDetector(),
            PhoneDetector(),
            CardNumDetector()
        ]
        print("✅ [RegexMatcher] Initialization complete.")

    def detect(self, text: str) -> List[Dict[str, Any]]:
        """
        주어진 텍스트에서 모든 PII를 탐지하여 리스트로 반환합니다.
        
        Args:
            text (str): 분석할 문장
            
        Returns:
            List[Dict]: [
                {
                    "start": int,
                    "end": int,
                    "match": str,       # 탐지된 문자열
                    "label": str,       # 전화번호, 주민번호 등
                    "score": float,     # 신뢰도 점수
                    "category": str,    # 개인/기밀
                    "type": str         # 식별/준식별
                }, ...
            ]
        """
        results = []

        for detector in self.detectors:
            # 각 디텍터의 detect 메서드 호출
            # (모든 디텍터는 BaseDetector를 상속받아 표준화된 결과를 반환함)
            matches = detector.detect(text)
            
            for m in matches:
                # 1. Match 문자열 추출 (BaseDetector가 대부분 처리해주지만 방어 로직 유지)
                if "match" not in m:
                    m["match"] = text[m["start"]:m["end"]]
                
                # 2. Score 계산 (이미 detect 내부에서 계산되지만, 없을 경우 fallback)
                if "score" not in m or m["score"] is None:
                    # BaseDetector의 score 메서드는 기본 1.0 반환
                    m["score"] = detector.score(m["match"])

                # 3. 메타 정보 매핑 (개인/기밀, 식별/준식별)
                label = m["label"]
                mapping = self.DETECTOR_TYPE_MAP.get(label, {"category": "Unknown", "type": "Unknown"})

                # 4. 결과 포맷팅
                result_item = {
                    "start": m["start"],
                    "end": m["end"],
                    "match": m["match"],
                    "label": label,
                    "score": float(m["score"]),
                    "category": mapping["category"], # 개인/기밀
                    "type": mapping["type"]          # 식별/준식별
                }
                
                results.append(result_item)

        # 시작 위치 순으로 정렬 (가독성 및 후처리 편의를 위해)
        results.sort(key=lambda x: x["start"])
        return results