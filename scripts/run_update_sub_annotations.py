# scripts/run_update_sub_annotations.py

import os
import json
import yaml
import re
from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.database.crud import get_dtm_by_domain
from tqdm import tqdm
import pandas as pd

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
    z_score를 비교하여 적절한 접미사 반환 (예: "_3", "_2", "_1")
    """
    if z_score is None:
        return ""
    
    for entry in thresholds:
        if z_score >= entry['threshold']:
            return entry['suffix']
            
    return default_suffix

def remove_existing_label_suffix(label, thresholds, default_suffix):
    """
    label 명 뒤에 붙은 기존 접미사를 제거합니다.
    예: "개인정보_3" -> "개인정보"
    """
    # # 설정에 정의된 모든 접미사 패턴 추출
    # all_suffixes = [t['suffix'] for t in thresholds] + [default_suffix]
    
    # # 정규표현식: ( _3|_2|_1)$ 매칭
    # suffix_pattern = f"({'|'.join(map(re.escape, all_suffixes))})$"
    
    # return re.sub(suffix_pattern, "", label)
    return label.split('_')[0]

def get_all_ground_truth(train_data_root, domain_dir):
    domain_path = os.path.join(train_data_root, domain_dir)
    json_files = [f for f in os.listdir(domain_path) if f.endswith(".json")]

    curr_df_dict = {
        'sentence': [],
        'sentence_id': [],
        'gt_label': [],
        'word': [],
        'start': [],
        'end': []
    }

    for json_file in tqdm(json_files, desc=f"GT 추출 중"):
        file_path = os.path.join(domain_path, json_file)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)

        for item in doc_data.get('data', []):
            curr_item_sentence = item['sentence']
            curr_item_id = item['id']
            for anno in item.get('annotations', []):
                curr_anno_word = anno['word']
                curr_anno_start = anno['start']
                curr_anno_end = anno['end']
                curr_anno_label = anno['label']

                curr_df_dict['sentence'].append( curr_item_sentence )
                curr_df_dict['sentence_id'].append( curr_item_id )
                curr_df_dict['gt_label'].append( curr_anno_label )
                curr_df_dict['word'].append( curr_anno_word )
                curr_df_dict['start'].append( curr_anno_start )
                curr_df_dict['end'].append( curr_anno_end )
        
    curr_df = pd.DataFrame(curr_df_dict)

    return curr_df


def run():
    # 1. 설정 로드
    try:
        z_config = load_z_score_config()
    except Exception as e:
        print(f"[Error] 설정 로드 실패: {e}")
        return

    edit_cfg = z_config['annotation_edit']
    thresholds = sorted(edit_cfg['thresholds'], key=lambda x: x['threshold'], reverse=True)
    default_suffix = edit_cfg['default_suffix']

    train_data_root = "data/train_data"
    db: Session = SessionLocal()

    # GT값 모으는 DF list
    dfs = []
    
    try:
        # 2. 도메인 디렉토리 순회
        domain_dirs = [d for d in os.listdir(train_data_root) 
                       if os.path.isdir(os.path.join(train_data_root, d))]
        
        for domain_dir in domain_dirs:
            try:
                domain_id = int(domain_dir.split('_')[0])
            except (ValueError, IndexError):
                continue

            print(f"\n>>> [도메인 {domain_id}] 라벨 업데이트 시작")

            # 3. 해당 도메인의 DTM(단어-점수 매핑) 로드
            dtm_map = {}
            for row in get_dtm_by_domain(db, domain_id):
                dtm_map[row['term']] = row['z_score']

            if not dtm_map:
                print(f"[-] DTM 데이터 없음: 도메인 {domain_id}")
                continue

            domain_path = os.path.join(train_data_root, domain_dir)
            json_files = [f for f in os.listdir(domain_path) if f.endswith(".json")]

            # 4. JSON 파일 수정
            for json_file in tqdm(json_files, desc=f"라벨 수정 중"):
                file_path = os.path.join(domain_path, json_file)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc_data = json.load(f)

                is_file_modified = False
                
                for item in doc_data.get('data', []):
                    for anno in item.get('annotations', []):
                        word = anno['word']    # 점수 조회는 '단어'로
                        current_label = anno['label'] # 수정 대상은 '라벨'
                        
                        # A. 기존 라벨에서 접미사 제거 (예: "개인정보_3" -> "개인정보")
                        clean_label = remove_existing_label_suffix(current_label, thresholds, default_suffix)
                        
                        # B. DB에서 단어(word) 기준으로 z_score 조회
                        z_val = dtm_map.get(word)
                        
                        # C. 새로운 접미사 결정
                        new_suffix = get_suffix(z_val, thresholds, default_suffix)
                        
                        if new_suffix:
                            updated_label = f"{clean_label}{new_suffix}"
                            
                            anno['label'] = updated_label
                            is_file_modified = True

                # 5. 저장
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(doc_data, f, ensure_ascii=False, indent=4)

            curr_df = get_all_ground_truth(train_data_root, domain_dir)
            dfs.append( curr_df )

        final_df = pd.concat(dfs, ignore_index=True)
        log_path = os.path.join(train_data_root, 'gt_label_log.csv')
        final_df.to_csv(log_path, index=False, encoding='utf-8-sig')

    except Exception as e:
        print(f"[Error] 실행 중 오류: {e}")
    finally:
        db.close()
        print("\n[완료] 모든 데이터의 라벨 접미사 처리가 끝났습니다.")

if __name__ == "__main__":
    run()