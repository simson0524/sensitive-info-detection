# src/database/models.py

from sqlalchemy import Column, String, Integer, BigInteger, Float, Text, ForeignKey, DateTime, ForeignKeyConstraint, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.database.connection import Base

# --------------------------------------------------------
# 1. 실험 전체 개요 (Experiment)
# --------------------------------------------------------
class Experiment(Base):
    __tablename__ = "experiment"

    experiment_code = Column(Text, primary_key=True)
    previous_experiment_code = Column(Text)
    data_category = Column(Text)
    run_mode = Column(Text)
    experiment_config = Column(JSONB)
    dataset_absolute_path = Column(Text)
    dataset_info = Column(JSONB)
    
    experiment_start_time = Column(DateTime(timezone=True))
    experiment_end_time = Column(DateTime(timezone=True))
    experiment_duration = Column(Float) # DOUBLE PRECISION

    # 관계 설정 (ORM 편의성용)
    process_results = relationship("ExperimentProcessResult", back_populates="experiment", cascade="all, delete-orphan")


# --------------------------------------------------------
# 2. 각 프로세스별 개요 (Process Results)
# --------------------------------------------------------
class ExperimentProcessResult(Base):
    __tablename__ = "experiment_process_results"

    # 복합 Primary Key 구성 (PK1, PK2, PK3)
    experiment_code = Column(Text, ForeignKey("experiment.experiment_code", ondelete="CASCADE"), primary_key=True)
    process_code = Column(Text, primary_key=True)
    process_epoch = Column(Integer, primary_key=True)

    process_start_time = Column(DateTime(timezone=True))
    process_end_time = Column(DateTime(timezone=True))
    process_duration = Column(Float)
    process_results = Column(JSONB)

    # 관계 설정
    experiment = relationship("Experiment", back_populates="process_results")
    inference_sentences = relationship("ExperimentProcessInferenceSentence", back_populates="process_result", cascade="all, delete-orphan")


# --------------------------------------------------------
# 3. 각 프로세스별 문장 단위 추론 정보 (Inference Sentences)
# --------------------------------------------------------
class ExperimentProcessInferenceSentence(Base):
    __tablename__ = "experiment_process_inference_sentences"

    id = Column(BigInteger, primary_key=True, autoincrement=True) # PK (BIGINT)
    
    # 복합 외래 키 (FK1, FK2, FK3) - 부모 테이블의 PK 3개를 참조
    experiment_code = Column(Text, nullable=False)
    process_code = Column(Text, nullable=False)
    process_epoch = Column(Integer, nullable=False)
    sentence_id = Column(Text, nullable=False)
    sentence_inference_result = Column(JSONB)
    create_dtm = Column(DateTime(timezone=True), default=func.now())

    # 복합 FK 설정 (중요!)
    __table_args__ = (
        ForeignKeyConstraint(
            ['experiment_code', 'process_code', 'process_epoch'],
            ['experiment_process_results.experiment_code', 'experiment_process_results.process_code', 'experiment_process_results.process_epoch'],
            ondelete='CASCADE'
        ),
    )

    # 관계 설정
    process_result = relationship("ExperimentProcessResult", back_populates="inference_sentences")


# --------------------------------------------------------
# 4. 각 도메인 별 개인/기밀정보 사전 (Info Dictionary)
# --------------------------------------------------------
class InfoDictionary(Base):
    __tablename__ = "info_dictionary"

    # 복합 Primary Key (PK1, PK2, PK3)
    annotated_word = Column(Text, primary_key=True)
    data_category = Column(Text, primary_key=True)
    domain_id = Column(Text, primary_key=True)

    z_score_of_the_word = Column(JSONB)
    first_inserted_experiment_code = Column(Text)
    insertion_counts = Column(Integer)
    deletion_counts = Column(Integer)

    # 관계 설정
    sentences = relationship("InfoDictionarySentence", back_populates="dictionary_word", cascade="all, delete-orphan")


# --------------------------------------------------------
# 5. 사전에 등재된 단어가 포함된 문장 (Dictionary Sentences)
# --------------------------------------------------------
class InfoDictionarySentence(Base):
    __tablename__ = "info_dictionary_sentences"

    id = Column(BigInteger, primary_key=True, autoincrement=True) # PK (BIGINT)

    # 복합 외래 키 (FK1, FK2, FK3)
    annotated_word = Column(Text, nullable=False)
    data_category = Column(Text, nullable=False)
    domain_id = Column(Text, nullable=False)

    origin_sentence = Column(Text)

    # 복합 FK 설정
    __table_args__ = (
        ForeignKeyConstraint(
            ['annotated_word', 'data_category', 'domain_id'],
            ['info_dictionary.annotated_word', 'info_dictionary.data_category', 'info_dictionary.domain_id'],
            ondelete='CASCADE'
        ),
    )

    # 관계 설정
    dictionary_word = relationship("InfoDictionary", back_populates="sentences")


# --------------------------------------------------------
# 6. 신도메인 생성 프로세스 개요 (New Domain Gen Process)
# --------------------------------------------------------
class NewDomainDatasetGenerationProcess(Base):
    __tablename__ = "new_domain_dataset_generation_process"

    generated_domain_id = Column(Integer, primary_key=True)
    
    generation_start_time = Column(DateTime(timezone=True))
    generation_end_time = Column(DateTime(timezone=True))
    generation_duration = Column(Float)
    generation_config = Column(JSONB)
    generation_results = Column(JSONB)


# --------------------------------------------------------
# 7. 단어 출현 횟수의 Z-score 계산을 위한 TF-IDF 점수 테이블 (DTM)
# --------------------------------------------------------
class DomainTermMatrix(Base):
    __tablename__ = "domain_term_matrix"

    # 복합 Primary Key (PK1, PK2)
    domain_id = Column(
        Integer, 
        ForeignKey("domain.domain_id", ondelete="CASCADE"), 
        primary_key=True
    )
    term = Column(
        Text, 
        ForeignKey("term.term", ondelete="CASCADE"), 
        primary_key=True
    )

    tf_score = Column(Float)
    idf_score = Column(Float)
    tfidf_score = Column(Float)
    z_score = Column(Float)
    is_sensitive_label = Column(Boolean, default=False, nullable=False)

    # 관계 설정 (ORM 접근용)
    domain_info = relationship("Domain", back_populates="term_matrices")
    term_info = relationship("Term", back_populates="term_matrices")


# --------------------------------------------------------
# 8. 도메인 정보 및 도메인 count 함수를 이용한 총 도메인 수 추출을 위한 테이블
# --------------------------------------------------------
class Domain(Base):
    __tablename__ = "domain"

    domain_id = Column(Integer, primary_key=True)

    domain_name = Column(Text)

    # 관계 설정 (선택 사항: Domain 객체에서 바로 DTM 기록에 접근 가능)
    term_matrices = relationship("DomainTermMatrix", back_populates="domain_info", cascade="all, delete-orphan")


# --------------------------------------------------------
# 9. IDF 점수 계산을 위한 기준 단어가 포함된 도메인의 수 정보 저장 테이블
# --------------------------------------------------------
class Term(Base):
    __tablename__ = "term"

    term = Column(Text, primary_key=True)

    included_domain_counts = Column(Integer)
    avg_tfidf = Column(Float)
    stddev_tfidf = Column(Float)
    sum_tfidf = Column(Float)
    sum_square_tfidf = Column(Float)

    # 관계 설정 (선택 사항: Term 객체에서 이 단어가 쓰인 DTM 기록들에 접근 가능)
    term_matrices = relationship("DomainTermMatrix", back_populates="term_info", cascade="all, delete-orphan")


# --------------------------------------------------------
# 테이블 매핑 (CRUD 유틸리티용)
# --------------------------------------------------------
TABLE_MAPPING = {
    'experiment': Experiment,
    'experiment_process_results': ExperimentProcessResult,
    'experiment_process_inference_sentences': ExperimentProcessInferenceSentence,
    'info_dictionary': InfoDictionary,
    'info_dictionary_sentences': InfoDictionarySentence,
    'new_domain_dataset_generation_process': NewDomainDatasetGenerationProcess,
    'domain_term_matrix': DomainTermMatrix,
    'domain': Domain,
    'term': Term
}