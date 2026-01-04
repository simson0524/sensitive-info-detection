# scripts/run_update_sub_annotations.py

import os
import json
import yaml
import re
from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.database.crud import get_dtm_by_domain
from tqdm import tqdm

def load_z_score_config(config_path="configs/z_score_config.yaml"):
    """
    Z-score 임계값 및 접미사 설정 파일을 로드합니다.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_suffix(z_score, thresholds, default_suffix):
    """
    입력받은 z_score를 임계값과 비교하여 적절한 접미사를 반환합니다.
    """
    if z_score is None:
        return ""
    
    # 내림차순 정렬된 thresholds를 순회하며 조건 확인
    for entry in thresholds:
        if z_score >= entry['threshold']:
            return entry['suffix']
            
    return default_suffix

def remove_existing_suffix(word, thresholds, default_suffix):
    """
    단어 끝에 이미 붙어있을 수 있는 접미사(_1, _2, _3 등)를 제거합니다.
    설정 파일(yaml)에 정의된 모든 suffix 패턴을 찾아 삭제합니다.
    """
    # 설정에 정의된 모든 가능한 접미사 리스트 생성
    all_suffixes = [t['suffix'] for t in thresholds] + [default_suffix]
    
    # 정규표현식 패턴 생성: 예) "(_3|_2|_1)$" 
    # re.escape는 접미사에 특수문자가 있을 경우를 대비해 처리하며, $는 문자열 끝 매칭을 의미
    suffix_pattern = f"({'|'.join(map(re.escape, all_suffixes))})$"
    
    # 패턴에 매칭되는 접미사를 제거한 순수 단어 반환
    return re.sub(suffix_pattern, "", word)

def run():
    # 1. 설정값 로드 및 환경 준비
    try:
        z_config = load_z_score_config()
    except Exception as e:
        print(f"[Error] 설정을 로드하는 중 오류 발생: {e}")
        return

    edit_cfg = z_config['annotation_edit']
    # 임계값을 큰 순서대로 정렬 (로직 정확성 확보)
    thresholds = sorted(edit_cfg['thresholds'], key=lambda x: x['threshold'], reverse=True)
    default_suffix = edit_cfg['default_suffix']

    train_data_root = "data/train_data"
    db: Session = SessionLocal()
    
    try:
        # 2. train_data 폴더 내 도메인 디렉토리 목록 추출
        domain_dirs = [d for d in os.listdir(train_data_root) 
                       if os.path.isdir(os.path.join(train_data_root, d))]
        
        for domain_dir in domain_dirs:
            # 폴더명에서 domain_id 추출 (예: "002_medical" -> 2)
            try:
                domain_id_str = domain_dir.split('_')[0]
                domain_id = int(domain_id_str)
            except (ValueError, IndexError):
                print(f"[Warning] 폴더명 형식이 유효하지 않음: {domain_dir}")
                continue

            print(f"\n>>> [도메인 {domain_id}] 처리 중...")

            # 3. DB에서 해당 도메인의 DTM 데이터를 메모리에 로드 (속도 최적화)
            dtm_map = {}
            for row in get_dtm_by_domain(db, domain_id):
                dtm_map[row['term']] = row['z_score']

            if not dtm_map:
                print(f"[-] 도메인 {domain_id}의 DTM 데이터가 DB에 존재하지 않아 스킵합니다.")
                continue

            # 4. 도메인 디렉토리 내 JSON 파일 순회
            domain_path = os.path.join(train_data_root, domain_dir)
            json_files = [f for f in os.listdir(domain_path) if f.endswith(".json")]

            for json_file in tqdm(json_files, desc=f"JSON 업데이트 ({domain_id_str})"):
                file_path = os.path.join(domain_path, json_file)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc_data = json.load(f)

                is_file_modified = False
                
                # JSON 데이터 구조를 순회하며 'word' 업데이트
                for item in doc_data.get('data', []):
                    for anno in item.get('annotations', []):
                        current_word = anno['word']
                        
                        # A. 기존 접미사 제거 (중복 방지 및 정확한 DB 조회를 위함)
                        clean_word = remove_existing_suffix(current_word, thresholds, default_suffix)
                        
                        # B. DB 점수 맵에서 순수 단어(clean_word)로 Z-score 조회
                        z_val = dtm_map.get(clean_word)
                        
                        # C. 현재 설정값 기준으로 새로운 접미사 결정
                        new_suffix = get_suffix(z_val, thresholds, default_suffix)
                        
                        if new_suffix:
                            updated_word = f"{clean_word}{new_suffix}"
                            
                            # 실제 단어가 변경되었을 때만 업데이트 수행
                            if current_word != updated_word:
                                anno['word'] = updated_word
                                is_file_modified = True

                # 5. 변경 사항이 발생한 파일만 저장
                if is_file_modified:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(doc_data, f, ensure_ascii=False, indent=4)

    except Exception as e:
        print(f"[Fatal Error] 실행 중 예외 발생: {e}")
    finally:
        db.close()
        print("\n[알림] 작업이 완료되었습니다.")

if __name__ == "__main__":
    run()