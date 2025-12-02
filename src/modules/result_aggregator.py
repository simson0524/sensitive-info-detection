# src/modules/result_aggregator.py

from collections import defaultdict

class ResultAggregator:
    """
    검증 결과(정탐/오탐/미탐)를 수집하고 통계를 계산하는 클래스
    Process 2(사전), Process 3(정규식) 등에서 공통으로 사용 가능
    """
    def __init__(self):
        # { "hit": [], "wrong": [], "mismatch": [] }
        self.logs = defaultdict(list)
        # { label_id: {"hit": 0, "wrong": 0, "mismatch": 0} }
        self.metrics = defaultdict(lambda: {"hit": 0, "wrong": 0, "mismatch": 0})

    def add_result(self, result_type: str, log_data: dict, pred_label_id: int):
        """
        결과 추가
        :param result_type: "hit", "wrong", "mismatch"
        :param log_data: DB에 저장할 로그 데이터 딕셔너리
        :param pred_label_id: 예측된 라벨 ID (메트릭 집계용)
        """
        if result_type not in ["hit", "wrong", "mismatch"]:
            return
            
        self.logs[result_type].append(log_data)
        self.metrics[pred_label_id][result_type] += 1

    def get_logs(self, result_type: str) -> list:
        return self.logs[result_type]

    def get_metrics(self) -> dict:
        """
        집계된 메트릭 반환 (JSON serializable)
        """
        return dict(self.metrics)