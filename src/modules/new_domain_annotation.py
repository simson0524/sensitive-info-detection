# src/modules/new_domain_annotation.py

import os
import json
import pandas as pd
import zipfile
import shutil
import re  # 정규표현식을 사용하여 문장 내 모든 패턴(단어) 위치를 찾기 위해 사용
from tqdm import tqdm

def new_domain_annotation(temp_dir, domain_id, domain_name, zip_storage_dir):
    """
    [데이터 파이프라인 3단계: 정규표현식 기반 전수 매핑 및 최종 패키징]
    
    기능:
    1. 생성된 JSON 문장들에서 정답지(CSV)에 기록된 단어가 나타나는 '모든' 위치를 탐색.
    2. 중복된 위치나 중복된 레이블이 주입되지 않도록 필터링.
    3. 어노테이션이 완료된 파일들을 하나로 묶어 최종 ZIP 파일 생성 및 임시 파일 삭제.
    """
    
    # --- [준비 단계] 정답지(CSV) 로드 및 대상 JSON 목록 확보 ---
    csv_path = os.path.join(temp_dir, "answer_sheet.csv")
    if not os.path.exists(csv_path):
        print(f"[Skip] {domain_id} 도메인에 정답지(CSV)가 없어 어노테이션을 건너뜁니다.")
        return

    # Pandas를 이용해 CSV를 읽어와 효율적인 필터링 준비
    df_ans = pd.read_csv(csv_path)
    # 현재 임시 폴더에 있는 모든 JSON 파일 목록 추출
    json_files = [f for f in os.listdir(temp_dir) if f.endswith(".json")]

    # --- [1단계] 전수 매핑 루프 시작 ---
    for json_file in tqdm(json_files, desc=f"Full Mapping [{domain_id}]"):
        file_path = os.path.join(temp_dir, json_file)
        
        # 1-1. JSON 파일 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            doc_content = json.load(f)

        # 1-2. 현재 문서 ID를 기준으로 이 문서에 해당하는 정답 데이터만 필터링
        current_doc_id = doc_content["document_id"]
        doc_answers = df_ans[df_ans['document'] == current_doc_id]

        # 1-3. 문서 내 각 문장 객체를 순회하며 텍스트 매칭
        for sentence_obj in doc_content.get("data", []):
            s_id = sentence_obj["id"]         # 문장 ID
            s_text = sentence_obj["sentence"]  # 실제 문장 텍스트
            
            # 해당 문장 ID에 배정된 정답 정보(word, label)만 가져옴
            s_answers = doc_answers[doc_answers['sentence'] == s_id]
            
            # [중복 방지용 셋] 동일한 위치에 동일한 레이블이 중복 추가되는 것을 막기 위함
            existing_spans = set() 

            for _, row in s_answers.iterrows():
                # 데이터가 비어있는 경우(NaN) 예외 처리
                if pd.isna(row['word']): continue
                
                # 검색할 단어 양끝 공백 제거 및 문자열 강제 변환
                word = str(row['word']).strip()
                if not word: continue
                label = row['label']

                # [핵심 로직] re.finditer 활용
                # 1) re.escape(word): 단어에 정규표현식 기호(?, *, [ 등)가 있어도 순수 문자로 인식하게 함
                # 2) re.finditer: 문장 전체에서 해당 단어가 출현하는 '모든' 매치 객체를 반환
                matches = re.finditer(re.escape(word), s_text)

                for match in matches:
                    start_idx = match.start() # 단어 시작 인덱스
                    end_idx = match.end()     # 단어 끝 인덱스
                    
                    # 중복 주입을 방지하기 위한 고유 키 (시작, 끝, 레이블)
                    span_key = (start_idx, end_idx, label)
                    
                    if span_key not in existing_spans:
                        # 신규 위치라면 annotations 리스트에 규격대로 추가
                        sentence_obj["annotations"].append({
                            "word": word,
                            "start": start_idx,
                            "end": end_idx,
                            "label": label
                        })
                        # 기록을 남겨 다음 루프에서 중복되지 않게 함
                        existing_spans.add(span_key)

        # 1-4. 어노테이션이 완료된 데이터를 다시 JSON 파일로 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(doc_content, f, indent=4, ensure_ascii=False)

    # --- [2단계] 패키징: 결합 완료된 파일들을 ZIP으로 압축 ---
    # 저장될 파일명 결정 (예: 001_finance.zip)
    zip_name = f"{domain_id}_{domain_name}.zip"
    final_zip_path = os.path.join(zip_storage_dir, zip_name)
    
    # ZIP_DEFLATED 옵션으로 용량을 압축하여 저장
    with zipfile.ZipFile(final_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for file in os.listdir(temp_dir):
            # temp_dir 내의 JSON들과 answer_sheet.csv를 포함시킴
            z.write(os.path.join(temp_dir, file), file)

    # --- [3단계] 정리: 작업이 완료된 임시 폴더 삭제 ---
    shutil.rmtree(temp_dir)
    print(f"\n[Success] '{domain_id}_{domain_name}' 모든 출현 위치 매핑 및 압축 완료")