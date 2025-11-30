# 🛡️ Sensitive Info Detector
> **BERT 기반 민감정보(개인정보/기밀정보) 탐지 및 마스킹 프로젝트**

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Database Schema](#-database-schema-erd)
- [Installation](#-installation)

## 🧐 Project Overview

이 프로젝트는 텍스트 내에 포함된 개인정보(PI Information) & 기업 기밀정보(Confidential Information)를 탐지하여...
(설명 내용)

## 🗂️ Database Schema (ERD)
> 프로젝트에서 사용된 데이터베이스 구조는 아래와 같습니다.

![ER Diagram](./assets/ERD251201.jpg)

## 📁 Project Structure
> 프로젝트에서 사용된 소스코드 및 파일들의 구조입니다.

sensitive-info-detector/
├── assets/  # 프로젝트 관련 이미지 관리 ✅
│
├── configs/ ✅
│   ├── base_config.yaml        # 기본 설정을 관리하는 yaml파일 ✅
│   └── experiment_config.yaml  # 실험별 하이퍼파라미터를 관리하는 파일 ✅
│
├── data/ ✅
│   ├── zip_raw_data/  # 도메인별로 관리되는 압축폴더(원본 유지용 & 데이터 수정 절대 불가능)
│   │   └── {domain_id}_{domain_name}.zip
│   │
│   └── train_data/  # 도메인별로 관리되는 폴더(데이터와 정답지 포함 & 필요에 따라 데이터 수정 가능)
│       └── {domain_id}_{domain_name}
│           ├── {domain_id}_{document_id}.json  # 문서단위 데이터
│           ├── ...                 # 문서단위 데이터
│           └── answer_sheet.csv    # 정답지(단순 추론 대상 도메인이라면 없음)
│
├── outputs/  # 실험 결과 및 로그 ✅
│   ├── checkpoints/  # 학습 모델 가중치
│   │   └── {experiment_code}_{process_epoch}.pt
│   │
│   └── logs/  # 실험단위로 관리되는 실험 로그
│       └── {experiment_code}/
│           ├── {experiment_code}_{process_code}_{process_epoch}_inference_sentences.csv  # 각 프로세스에서 문장 단위 추론 결과
│           ├── {experiment_code}_all_process_results.txt                                 # 실험 + 모든 프로세스의 결과를 순서대로 작성한 txt
│           ├── {experiment_code}_loss_graph.png                                          # 모델 학습 중 train & valid loss 추이를 나타낸 그래프
│           ├── {experiment_code}_label_count_graph.png                                   # 모델 학습 중 정탐오탐미탐 샘플 수 추이를 나타낸 그래프
│           └── {experiment_code}_experiment_log.txt                                      # 실험 파이프라인 실행 중 발생하는 모든 print log
│   
├── src/  # 소스 코드 (Package)
│   ├── database/  # DB 관련 로직 (ERD 기반) ✅
│   │   ├── init_db.py     # DB 초기화 로직 (동일 config DB가 있는 경우에도 안전하게 실행 가능) ✅
│   │   ├── config.py      # DB 접속 정보 및 URL 설정 관리 ✅
│   │   ├── connection.py  # DB 세션(Session) 생명주기 및 연결 관리 ✅
│   │   ├── models.py      # ORM 모델 클래스 및 테이블 스키마 정의 ✅
│   │   └── crud.py        # 데이터 IO(CRUD) 로직 및 타입 변환 (ORM↔Dict) ✅
│   │
│   ├── models/  # 모델 아키텍처
│   │   └── ner_roberta.py ✅
│   │
│   ├── modules/  # 각 탐지 로직의 핵심 모듈
│   │   ├── ner_preprocessor.py    # [NER모델]데이터 전처리 및 로드 모듈 ✅
│   │   ├── ner_trainer.py         # [NER모델]학습 모듈 ✅
│   │   ├── ner_evaluator.py       # [NER모델]검증 모듈 ✅
│   │   ├── dictionary_matcher.py  # 사전 매칭 모듈 ✅
│   │   ├── dictionary_updater.py  # 사전 업데이트 모듈
│   │   ├── regex_matcher.py       # 정규표현식 매칭 모듈 ✅
│   │   └── regex_logics/          # 정규표현식 매칭 모듈에서 사용하는 로직 ✅
│   │
│   ├── processes/  # 실행 프로세스 
│   │   ├── process_1.py  # 모델학습 및 검증 프로세스        
│   │   ├── process_2.py  # 사전 매칭 검증 프로세스
│   │   ├── process_3.py  # 정규표현식 매칭 검증 프로세스         
│   │   └── process_4.py  # 모델 검증 프로세스
│   │
│   └── utils/  # 유틸리티
│       ├── common.py      # YAML 로드, 시드 고정, 디렉토리 생성 등 공통 함수
│       ├── metrics.py     # 평가 지표 계산
│       └── visualizer.py  # 그래프그리는 친구
│
├── tools/  # 실행과 별개인 도구들 (labeling_tools)
│   ├── candidate_labeler.py
│   ├── manual_validator.py
│   └── metric_viewer.py
│
├── scripts/                     # 실제 실행 진입점 (Entry Points)
│   ├── init_project.py          # DB 생성 및 초기 사전 구축 (create_dbs + init_dictionary)
│   ├── run_experiment.py        # (run_pipeline.py) 실험 전체 파이프라인 실행
│   └── run_augmentation.py      # 증강만 따로 돌릴 때
│
├── .env                         # DB 접속 정보, 비밀키
├── .gitignore
│
├── README.md
└── requirements.txt