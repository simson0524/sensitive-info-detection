# src/modules/regex_logics/detectors/base.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseDetector(ABC):
    """
    모든 탐지기(Detector)가 상속받아야 하는 추상 기본 클래스
    """

    @abstractmethod
    def detect(self, text: str) -> List[Dict[str, Any]]:
        """
        텍스트에서 특정 패턴을 탐지하여 결과 리스트를 반환합니다.

        Args:
            text (str): 분석할 입력 문장

        Returns:
            List[Dict[str, Any]]: 탐지된 결과 리스트
            
            [예시 포맷]
            [
                {
                    "start": 10,
                    "end": 23,
                    "match": "010-1234-5678",
                    "label": "전화번호",
                    "score": 1.0 (선택 사항, 없으면 score 메서드 사용됨)
                },
                ...
            ]
        """
        pass

    def score(self, match: str) -> float:
        """
        매칭된 문자열의 신뢰도(Confidence Score)를 계산합니다.
        
        기본값은 1.0 (확실함)입니다. 
        규칙이 복잡하거나 오탐 가능성이 있는 디텍터는 이 메서드를 오버라이딩해서 사용하세요.
        
        Args:
            match (str): 탐지된 문자열

        Returns:
            float: 0.0 ~ 1.0 사이의 점수
        """
        return 1.0