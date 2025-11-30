# src/modules/dictionary_matcher.py

from sqlalchemy.orm import Session
from src.database import crud

class DictionaryMatcher:
    def __init__(self, session: Session):
        """
        DictionaryMatcher 초기화
        :param session: DB 세션 (crud 호출용)
        """
        self.session = session
        # 메모리 캐시: { domain_id: { word: word_info_dict, ... }, ... }
        self._dictionaries = {}
        # 통계 정보: { domain_id: active_word_count }
        self._stats = {}

    def load_dictionaries(self, domain_ids: list, data_category: str):
        """
        DB에서 특정 카테고리와 도메인들의 사전 데이터를 메모리에 로드합니다.
        기존 'dictionary_size_calculator'의 로직을 포함합니다.
        
        :param domain_ids: 로드할 도메인 ID 리스트 (예: ['finance', 'medical'])
        :param data_category: 'PII' 또는 'CONFIDENTIAL' 등 (ERD의 data_category)
        """
        self._dictionaries = {}
        self._stats = {}

        print(f"📖 [DictionaryMatcher] '{data_category}' 사전 로딩 중... (Domains: {domain_ids})")

        for domain_id in domain_ids:
            self._dictionaries[domain_id] = {}
            active_count = 0
            
            # crud의 Generator를 통해 스트리밍 방식으로 가져옴
            iterator = crud.get_dictionary_by_category_and_domain(
                self.session, 
                data_category=data_category, 
                domain_id=domain_id
            )

            for row in iterator:
                word = row['annotated_word']
                
                # [기존 로직 계승]
                # insertion_counts > deletion_counts 인 경우에만 유효한 단어로 인정
                if row['insertion_counts'] > row['deletion_counts']:
                    self._dictionaries[domain_id][word] = row
                    active_count += 1
                else:
                    # 삭제된 단어는 로드하지 않음 (디버깅용 출력 가능)
                    pass
            
            # 통계 저장 (epsilon 처리 등은 필요하다면 여기서 수행)
            self._stats[domain_id] = max(active_count, 0.000001) # 0이면 epsilon
            
        print(f"✅ [DictionaryMatcher] 로드 완료. Stats: {self._stats}")

    def match(self, token: str, domain_id: str) -> bool:
        """
        특정 토큰이 해당 도메인의 사전에 존재하는지 확인합니다.
        
        :param token: 검사할 단어 (span_token)
        :param domain_id: 현재 문장의 도메인 ID
        :return: True(정탐/오탐 후보) / False(미탐 후보)
        """
        # 해당 도메인 사전이 로드되어 있지 않으면 False
        if domain_id not in self._dictionaries:
            return False
            
        # 단어가 사전에 있는지 확인 (O(1) Lookup)
        return token in self._dictionaries[domain_id]

    def get_stats(self) -> dict:
        """
        로드된 사전의 크기(유효 단어 수) 정보를 반환합니다.
        기존 'dictionary_size_calculator'의 반환값인 each_dict_size와 대응됩니다.
        """
        return self._stats

    def get_word_info(self, token: str, domain_id: str) -> dict:
        """
        매칭된 단어의 상세 정보(z-score 등)가 필요할 때 사용합니다.
        """
        if self.match(token, domain_id):
            return self._dictionaries[domain_id][token]
        return None