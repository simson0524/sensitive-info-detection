# src/database/crud.py

from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
from src.database.models import (
    Experiment, 
    ExperimentProcessResult, 
    ExperimentProcessInferenceSentence,
    InfoDictionary, 
    InfoDictionarySentence, 
    NewDomainDatasetGenerationProcess,
    DomainTermMatrix,
    Domain,
    Term
)

# ==============================================================================
# 0. 유틸리티 (ORM -> Dict 변환)
# ==============================================================================

def row_to_dict(row) -> dict:
    """
    SQLAlchemy ORM 객체를 순수 Python 딕셔너리로 변환합니다.
    
    Args:
        row: SQLAlchemy 모델 인스턴스 (단일 행)
        
    Returns:
        dict: 컬럼명을 키로 갖는 딕셔너리. row가 None이면 None 반환.
    """
    if row is None:
        return None
    return {c.name: getattr(row, c.name) for c in row.__table__.columns}


def rows_to_list(rows) -> list:
    """
    SQLAlchemy ORM 객체 리스트를 딕셔너리 리스트로 변환합니다.
    
    Args:
        rows: SQLAlchemy 모델 인스턴스들의 리스트
        
    Returns:
        list[dict]: 딕셔너리로 변환된 데이터 리스트
    """
    return [row_to_dict(row) for row in rows]


# ==============================================================================
# 1. 실험 전체 개요 (Experiment)
# ==============================================================================

def create_experiment(session: Session, data: dict) -> dict:
    """
    새로운 실험(Experiment) 정보를 생성합니다.
    
    Args:
        session: DB 세션
        data (dict): 실험 정보 딕셔너리 (예: {'experiment_code': 'EXP01', ...})
        
    Returns:
        dict: DB에 저장된 확정된 실험 정보 (Default 값 등이 포함됨)
    """
    db_obj = Experiment(**data)
    session.add(db_obj)
    session.flush()  # DB에 전송하여 제약조건 확인 및 Default값 생성
    session.refresh(db_obj) # DB에서 생성된 데이터를 객체에 반영
    return row_to_dict(db_obj)


def get_experiment(session: Session, experiment_code: str = None):
    """
    실험 정보를 조회합니다.
    
    Args:
        session: DB 세션
        experiment_code (str, optional): 조회할 실험 코드. 
            - None일 경우: 모든 실험 목록을 최신순으로 반환
            - 값일 경우: 해당 실험의 상세 정보 반환
            
    Returns:
        list[dict] OR dict: 실험 목록 또는 단일 실험 정보
    """
    query = session.query(Experiment)
    
    if experiment_code is None:
        rows = query.order_by(desc(Experiment.experiment_start_time)).all()
        return rows_to_list(rows)
    else:
        row = query.filter(Experiment.experiment_code == experiment_code).first()
        return row_to_dict(row)


def update_experiment(session: Session, experiment_code: str, update_data: dict) -> bool:
    """
    특정 실험의 정보를 수정합니다.
    
    Args:
        session: DB 세션
        experiment_code (str): 수정할 실험 코드
        update_data (dict): 수정할 컬럼과 값 (예: {'experiment_end_time': '...'})
        
    Returns:
        bool: 수정 성공 여부 (대상 행이 있었으면 True)
    """
    updated_count = session.query(Experiment).filter(
        Experiment.experiment_code == experiment_code
    ).update(update_data)
    return updated_count > 0


def delete_experiment(session: Session, experiment_code: str) -> bool:
    """
    특정 실험을 삭제합니다.
    주의: CASCADE 설정에 의해 하위 로그(ProcessResults, Sentences 등)도 함께 삭제됩니다.
    
    Args:
        session: DB 세션
        experiment_code (str): 삭제할 실험 코드
        
    Returns:
        bool: 삭제 성공 여부
    """
    deleted_count = session.query(Experiment).filter(
        Experiment.experiment_code == experiment_code
    ).delete()
    return deleted_count > 0


# ==============================================================================
# 2. 각 프로세스별 개요 (ExperimentProcessResult)
# ==============================================================================

def create_process_result(session: Session, data: dict) -> dict:
    """
    프로세스 결과 로그를 생성합니다.
    
    Args:
        session: DB 세션
        data (dict): 프로세스 정보 (experiment_code, process_code, process_epoch 필수)
        
    Returns:
        dict: 저장된 프로세스 결과 정보
    """
    db_obj = ExperimentProcessResult(**data)
    session.add(db_obj)
    session.flush()
    session.refresh(db_obj)
    return row_to_dict(db_obj)


def get_process_results(session: Session, experiment_code: str) -> list:
    """
    특정 실험에 속한 모든 프로세스 결과를 Epoch 순서대로 조회합니다.
    
    Args:
        session: DB 세션
        experiment_code (str): 실험 코드
        
    Returns:
        list[dict]: 프로세스 결과 리스트
    """
    rows = session.query(ExperimentProcessResult).filter(
        ExperimentProcessResult.experiment_code == experiment_code
    ).order_by(ExperimentProcessResult.process_epoch).all()
    return rows_to_list(rows)


def get_specific_process_result(session: Session, experiment_code: str, process_code: str, process_epoch: int) -> dict:
    """
    특정 실험-프로세스-에폭(복합키)에 해당하는 단일 결과를 조회합니다.
    
    Args:
        session: DB 세션
        experiment_code (str): 실험 코드
        process_code (str): 프로세스 코드 (예: 'train', 'eval')
        process_epoch (int): 에폭 번호
        
    Returns:
        dict: 단일 프로세스 결과 정보
    """
    row = session.query(ExperimentProcessResult).filter(
        and_(
            ExperimentProcessResult.experiment_code == experiment_code,
            ExperimentProcessResult.process_code == process_code,
            ExperimentProcessResult.process_epoch == process_epoch
        )
    ).first()
    return row_to_dict(row)


def update_process_result(session: Session, experiment_code: str, process_code: str, process_epoch: int, update_data: dict) -> bool:
    """
    특정 프로세스 단계의 정보를 수정합니다.
    
    Args:
        session: DB 세션
        experiment_code, process_code, process_epoch: 식별자
        update_data (dict): 수정할 데이터
        
    Returns:
        bool: 수정 성공 여부
    """
    updated_count = session.query(ExperimentProcessResult).filter(
        and_(
            ExperimentProcessResult.experiment_code == experiment_code,
            ExperimentProcessResult.process_code == process_code,
            ExperimentProcessResult.process_epoch == process_epoch
        )
    ).update(update_data)
    return updated_count > 0


def delete_process_result(session: Session, experiment_code: str, process_code: str, process_epoch: int) -> bool:
    """
    특정 프로세스 결과 로그를 삭제합니다.
    
    Args:
        session: DB 세션
        experiment_code, process_code, process_epoch: 식별자
        
    Returns:
        bool: 삭제 성공 여부
    """
    deleted_count = session.query(ExperimentProcessResult).filter(
        and_(
            ExperimentProcessResult.experiment_code == experiment_code,
            ExperimentProcessResult.process_code == process_code,
            ExperimentProcessResult.process_epoch == process_epoch
        )
    ).delete()
    return deleted_count > 0


# ==============================================================================
# 3. 각 프로세스별 문장 단위 추론 정보 (ExperimentProcessInferenceSentence)
# ==============================================================================

def bulk_insert_inference_sentences(session: Session, data_list: list[dict]):
    """
    대량의 추론 문장 데이터를 한 번에 삽입합니다 (Bulk Insert).
    성능 최적화를 위해 ORM 객체 생성 과정을 생략합니다.
    
    Args:
        session: DB 세션
        data_list (list[dict]): 삽입할 데이터 딕셔너리 리스트
    """
    if data_list:
        session.bulk_insert_mappings(ExperimentProcessInferenceSentence, data_list)


def get_inference_sentences(session: Session, experiment_code: str, process_code: str, process_epoch: int, batch_size: int = 1000):
    """
    특정 프로세스 단계의 모든 문장 로그를 조회합니다 (Streaming).
    대량 데이터를 처리하기 위해 yield를 사용하여 메모리를 절약합니다.
    
    Args:
        session: DB 세션
        experiment_code, process_code, process_epoch: 조회 조건
        batch_size (int): DB에서 한 번에 가져올 행의 개수 (기본 1000)
        
    Yields:
        dict: 문장 로그 딕셔너리 하나씩 반환
    """
    query = session.query(ExperimentProcessInferenceSentence).filter(
        and_(
            ExperimentProcessInferenceSentence.experiment_code == experiment_code,
            ExperimentProcessInferenceSentence.process_code == process_code,
            ExperimentProcessInferenceSentence.process_epoch == process_epoch
        )
    )
    # yield_per를 사용하면 fetchall()과 달리 데이터를 배치 단위로 가져옵니다.
    for row in query.yield_per(batch_size):
        yield row_to_dict(row)


def delete_inference_sentence(session: Session, sentence_id: int) -> bool:
    """
    특정 추론 문장 로그 하나를 삭제합니다 (ID 기반).
    
    Args:
        session: DB 세션
        sentence_id (int): 삭제할 문장의 PK ID
        
    Returns:
        bool: 삭제 성공 여부
    """
    deleted_count = session.query(ExperimentProcessInferenceSentence).filter(
        ExperimentProcessInferenceSentence.id == sentence_id
    ).delete()
    return deleted_count > 0


def delete_inference_sentences_by_process(session: Session, experiment_code: str, process_code: str) -> bool:
    """
    특정 프로세스에서 발생한 모든 문장 로그를 삭제합니다. (초기화용)
    
    Args:
        session: DB 세션
        experiment_code, process_code: 삭제할 프로세스 범위
        
    Returns:
        bool: 삭제 성공 여부 (삭제된 행이 있으면 True)
    """
    deleted_count = session.query(ExperimentProcessInferenceSentence).filter(
        and_(
            ExperimentProcessInferenceSentence.experiment_code == experiment_code,
            ExperimentProcessInferenceSentence.process_code == process_code
        )
    ).delete()
    return deleted_count > 0


# ==============================================================================
# 4. 각 도메인 별 개인/기밀정보 사전 (InfoDictionary)
# ==============================================================================

def create_dictionary_item(session: Session, data: dict) -> dict:
    """
    사전에 단어 하나를 추가합니다.
    Merge 전략 사용: 이미 존재하는 단어(PK 일치)라면 업데이트하고, 없으면 생성합니다.
    
    Args:
        session: DB 세션
        data (dict): 사전 데이터 정보
        
    Returns:
        dict: DB에 반영된 실제 데이터를 반환
    """
    db_obj = InfoDictionary(**data)
    
    merged_obj = session.merge(db_obj) 
    session.flush()
    
    return row_to_dict(merged_obj)


def bulk_insert_dictionary_items(session: Session, data_list: list[dict]):
    """
    사전 데이터를 대량으로 삽입합니다 (Bulk Insert).
    초기 데이터 구축 시 유용합니다.
    
    Args:
        session: DB 세션
        data_list (list[dict]): 사전 데이터 리스트
    """
    if data_list:
        session.bulk_insert_mappings(InfoDictionary, data_list)


def get_dictionary_by_domain(session: Session, domain_id: str, batch_size: int = 1000):
    """
    특정 도메인의 모든 사전 단어를 조회합니다 (Streaming).
    
    Args:
        session: DB 세션
        domain_id (str): 도메인 ID
        batch_size (int): 배치 크기
        
    Yields:
        dict: 사전 단어 정보 딕셔너리
    """
    query = session.query(InfoDictionary).filter(
        InfoDictionary.domain_id == domain_id
    )
    for row in query.yield_per(batch_size):
        yield row_to_dict(row)


def get_dictionary_by_category_and_domain(session: Session, data_category: str, domain_id: str, batch_size: int = 1000):
    """
    카테고리와 도메인을 기준으로 사전을 조회합니다 (Streaming).
    
    Args:
        session: DB 세션
        data_category (str): 데이터 카테고리 (예: 'PII')
        domain_id (str): 도메인 ID
        
    Yields:
        dict: 사전 단어 정보 딕셔너리
    """
    query = session.query(InfoDictionary).filter(
        and_(
            InfoDictionary.data_category == data_category,
            InfoDictionary.domain_id == domain_id
        )
    )
    for row in query.yield_per(batch_size):
        yield row_to_dict(row)


def get_word_info(session: Session, annotated_word: str, data_category: str, domain_id: str) -> dict:
    """
    특정 단어 하나에 대한 상세 정보를 조회합니다.
    
    Args:
        session: DB 세션
        annotated_word, data_category, domain_id: 복합키 식별자
        
    Returns:
        dict: 단어 정보
    """
    row = session.query(InfoDictionary).filter(
        and_(
            InfoDictionary.annotated_word == annotated_word,
            InfoDictionary.data_category == data_category,
            InfoDictionary.domain_id == domain_id
        )
    ).first()
    return row_to_dict(row)


def update_dictionary_item(session: Session, annotated_word: str, data_category: str, domain_id: str, update_data: dict) -> bool:
    """
    사전의 특정 단어 정보를 수정합니다 (Single Update).
    
    Args:
        session: DB 세션
        annotated_word, data_category, domain_id: 식별자
        update_data (dict): 수정할 값
        
    Returns:
        bool: 수정 성공 여부
    """
    updated_count = session.query(InfoDictionary).filter(
        and_(
            InfoDictionary.annotated_word == annotated_word,
            InfoDictionary.data_category == data_category,
            InfoDictionary.domain_id == domain_id
        )
    ).update(update_data)
    return updated_count > 0


def bulk_update_dictionary_items(session: Session, update_data_list: list[dict]):
    """
    사전 데이터를 대량으로 수정합니다 (Bulk Update).
    
    Args:
        session: DB 세션
        update_data_list (list[dict]): 
            PK 3개('annotated_word', 'data_category', 'domain_id')와 
            수정할 필드가 반드시 포함된 딕셔너리 리스트.
    """
    if update_data_list:
        session.bulk_update_mappings(InfoDictionary, update_data_list)


def delete_dictionary_item(session: Session, annotated_word: str, data_category: str, domain_id: str) -> bool:
    """
    사전에서 특정 단어 하나를 삭제합니다.
    
    Returns:
        bool: 삭제 성공 여부
    """
    deleted_count = session.query(InfoDictionary).filter(
        and_(
            InfoDictionary.annotated_word == annotated_word,
            InfoDictionary.data_category == data_category,
            InfoDictionary.domain_id == domain_id
        )
    ).delete()
    return deleted_count > 0


def invalidate_dictionary_item(session: Session, annotated_word: str, data_category: str, domain_id: str):
    """
    사전에서 특정 단어를 즉시 무효화합니다. (오탐 발생 시)
    Logic: deletion_counts를 insertion_counts와 동일하게 설정하여, 
           (insertion > deletion) 조건을 만족하지 못하게 만듭니다.
    """
    updated_count = session.query(InfoDictionary).filter(
        and_(
            InfoDictionary.annotated_word == annotated_word,
            InfoDictionary.data_category == data_category,
            InfoDictionary.domain_id == domain_id
        )
    ).update(
        # [핵심 변경] +1이 아니라, insertion_counts 값 그대로 복사
        {InfoDictionary.deletion_counts: InfoDictionary.insertion_counts},
        synchronize_session=False
    )
    return updated_count > 0


# ==============================================================================
# 5. 사전에 등재된 단어가 포함된 문장 (InfoDictionarySentence)
# ==============================================================================

def bulk_insert_dictionary_sentences(session: Session, data_list: list[dict]):
    """
    사전 단어 예시 문장들을 대량 삽입합니다 (Bulk Insert).
    
    Args:
        session: DB 세션
        data_list (list[dict]): 예시 문장 데이터 리스트
    """
    if data_list:
        session.bulk_insert_mappings(InfoDictionarySentence, data_list)


def get_sentences_by_word(session: Session, annotated_word: str, domain_id: str) -> list:
    """
    특정 단어가 포함된 예시 문장 목록을 조회합니다.
    
    Args:
        session: DB 세션
        annotated_word, domain_id: 검색 조건
        
    Returns:
        list[dict]: 예시 문장 리스트
    """
    rows = session.query(InfoDictionarySentence).filter(
        and_(
            InfoDictionarySentence.annotated_word == annotated_word,
            InfoDictionarySentence.domain_id == domain_id
        )
    ).all()
    return rows_to_list(rows)


def update_dictionary_sentence(session: Session, sentence_id: int, update_data: dict) -> bool:
    """
    특정 예시 문장의 내용을 ID 기반으로 수정합니다.
    """
    updated_count = session.query(InfoDictionarySentence).filter(
        InfoDictionarySentence.id == sentence_id
    ).update(update_data)
    return updated_count > 0


def delete_dictionary_sentence(session: Session, sentence_id: int) -> bool:
    """
    특정 예시 문장을 ID 기반으로 삭제합니다.
    """
    deleted_count = session.query(InfoDictionarySentence).filter(
        InfoDictionarySentence.id == sentence_id
    ).delete()
    return deleted_count > 0


# ==============================================================================
# 6. 신도메인 생성 프로세스 개요 (NewDomainDatasetGenerationProcess)
# ==============================================================================

def create_generation_process_log(session: Session, data: dict) -> dict:
    """
    신규 도메인 생성 로그를 저장합니다.
    
    Args:
        session: DB 세션
        data (dict): 생성 로그 정보
        
    Returns:
        dict: 저장된 로그 정보 (생성된 ID 포함)
    """
    db_obj = NewDomainDatasetGenerationProcess(**data)
    session.add(db_obj)
    session.flush()
    session.refresh(db_obj)
    return row_to_dict(db_obj)


def get_generation_process_log(session: Session, generated_domain_id: int) -> dict:
    """
    특정 생성 ID(PK)에 해당하는 로그를 조회합니다.
    
    Returns:
        dict: 로그 정보 (없으면 None)
    """
    row = session.query(NewDomainDatasetGenerationProcess).filter(
        NewDomainDatasetGenerationProcess.generated_domain_id == generated_domain_id
    ).first()
    return row_to_dict(row)


def update_generation_process_log(session: Session, generated_domain_id: int, update_data: dict) -> bool:
    """
    생성 로그 정보를 수정합니다.
    """
    updated_count = session.query(NewDomainDatasetGenerationProcess).filter(
        NewDomainDatasetGenerationProcess.generated_domain_id == generated_domain_id
    ).update(update_data)
    return updated_count > 0


def delete_generation_process_log(session: Session, generated_domain_id: int) -> bool:
    """
    특정 생성 로그를 삭제합니다.
    """
    deleted_count = session.query(NewDomainDatasetGenerationProcess).filter(
        NewDomainDatasetGenerationProcess.generated_domain_id == generated_domain_id
    ).delete()
    return deleted_count > 0


# ==============================================================================
# 7. 단어 출현 횟수 및 TF-IDF 점수 (DomainTermMatrix)
# ==============================================================================

def bulk_insert_dtm_items(session: Session, data_list: list[dict]):
    """
    DTM 데이터를 대량으로 '신규 삽입'합니다.
    (이미 데이터가 존재할 경우 IntegrityError가 발생하므로, 초기 생성 시에 사용합니다.)
    """
    if data_list:
        session.bulk_insert_mappings(DomainTermMatrix, data_list)


def bulk_update_dtm_items(session: Session, data_list: list[dict]):
    """
    기존 DTM 행들의 점수를 대량으로 '수정'합니다.
    Args:
        data_list: PK(domain_id, term)와 수정할 필드(tf_score 등)가 포함된 리스트
    """
    if data_list:
        session.bulk_update_mappings(DomainTermMatrix, data_list)


def get_dtm_by_domain(session: Session, domain_id: int, batch_size: int = 2000):
    """
    특정 도메인의 모든 단어 점수를 스트리밍 방식으로 조회합니다.
    
    Yields:
        dict: DTM 데이터 행
    """
    query = session.query(DomainTermMatrix).filter(DomainTermMatrix.domain_id == domain_id)
    for row in query.yield_per(batch_size):
        yield row_to_dict(row)


def get_dtm_by_term(session: Session, term: str) -> list[dict]:
    """
    특정 단어가 어느 도메인들에서 나타나는지 점수와 함께 조회합니다.
    """
    rows = session.query(DomainTermMatrix).filter(DomainTermMatrix.term == term).all()
    return rows_to_list(rows)


def delete_dtm_by_domain(session: Session, domain_id: int) -> int:
    """특정 도메인의 모든 DTM 데이터를 삭제합니다."""
    deleted_count = session.query(DomainTermMatrix).filter(
        DomainTermMatrix.domain_id == domain_id
    ).delete()
    return deleted_count


# ==============================================================================
# 8. 도메인 정보 (Domain)
# ==============================================================================

def create_domain(session: Session, domain_name: str) -> dict:
    """
    새로운 도메인을 생성하고 정보를 반환합니다.
    """
    db_obj = Domain(domain_name=domain_name)
    session.add(db_obj)
    session.flush()
    session.refresh(db_obj)
    return row_to_dict(db_obj)


def create_domain_with_id(session: Session, domain_id: int, domain_name: str) -> dict:
    """
    [추가] 특정 ID를 지정하여 도메인을 생성합니다. (dtm_initializer에서 사용)
    폴더명에서 추출한 고정 ID를 DB에 그대로 박아넣을 때 사용합니다.
    """
    db_obj = Domain(domain_id=domain_id, domain_name=domain_name)
    session.add(db_obj)
    session.flush()
    session.refresh(db_obj)
    return row_to_dict(db_obj)

def get_domain_by_name(session: Session, domain_name: str) -> dict:
    """
    [추가] 도메인 이름을 통해 정보를 조회합니다.
    """
    row = session.query(Domain).filter(Domain.domain_name == domain_name).first()
    return row_to_dict(row)


def get_all_domains(session: Session) -> list[dict]:
    """
    전체 도메인 목록을 조회합니다.
    """
    rows = session.query(Domain).all()
    return rows_to_list(rows)


def get_domain_count(session: Session) -> int:
    """
    총 도메인 수(IDF 계산의 분모)를 구합니다.
    """
    return session.query(Domain).count()


# ==============================================================================
# 9. 단어 통계 정보 (Term)
# ==============================================================================

def bulk_insert_terms(session: Session, data_list: list[dict]):
    """
    새로운 단어들을 통계 테이블에 대량 삽입합니다.
    """
    if data_list:
        session.bulk_insert_mappings(Term, data_list)


def bulk_update_terms(session: Session, data_list: list[dict]):
    """
    기존 단어들의 통계 정보(included_domain_counts 등)를 대량 수정합니다.
    Args:
        data_list: PK(term)와 수정할 필드가 포함된 리스트
    """
    if data_list:
        session.bulk_update_mappings(Term, data_list)


def get_term_stats(session: Session, term: str) -> dict:
    """
    특정 단어의 통계 정보(예: 몇 개의 도메인에 나타나는지)를 조회합니다.
    """
    row = session.query(Term).filter(Term.term == term).first()
    return row_to_dict(row)


def get_all_terms_streaming(session: Session, batch_size: int = 5000):
    """
    전체 단어장 정보를 스트리밍 조회합니다 (IDF 재계산 시 유용).
    """
    query = session.query(Term)
    for row in query.yield_per(batch_size):
        yield row_to_dict(row)