# temp

import os
import pandas as pd
from src.database.connection import db_manager
from src.database.models import DomainTermMatrix, Domain
from tqdm import tqdm

def extract_detailed_words():
    output_dir = "outputs/logs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "top_10_pearson_detailed_words.csv")
    
    # 1. 아까 생성한 상관관계 상세 CSV 로드
    summary_csv = os.path.join(output_dir, "correlation_details.csv")
    if not os.path.exists(summary_csv):
        print("❌ correlation_details.csv 파일이 없습니다. 먼저 분석 스크립트를 실행해주세요.")
        return

    df_summary = pd.read_csv(summary_csv)
    
    # 2. 피어슨 상관계수 상위 10개 추출 (id_a, id_b 기반)
    # 실제 DB 조회를 위해 ID가 필요하므로 id_a, id_b 컬럼을 사용합니다.
    top_10_pairs = df_summary.dropna(subset=['pearson']).nlargest(10, 'pearson')
    
    all_detailed_data = []

    print(f"🚀 상위 10개 조합에 대해 단어 추출을 시작합니다... (각 조합당 100개)")

    with db_manager.get_db() as session:
        for _, row in tqdm(top_10_pairs.iterrows(), total=10):
            id_a, id_b = int(row['id_a']), int(row['id_b'])
            name_a, name_b = row['domain_a_kor'], row['domain_b_kor']
            pearson_val = row['pearson']

            # 두 도메인의 단어 데이터 쿼리
            df_a = pd.read_sql(session.query(DomainTermMatrix.term, DomainTermMatrix.z_score).filter(DomainTermMatrix.domain_id == id_a).statement, session.bind)
            df_b = pd.read_sql(session.query(DomainTermMatrix.term, DomainTermMatrix.z_score).filter(DomainTermMatrix.domain_id == id_b).statement, session.bind)

            # 공통 단어 병합
            merged = pd.merge(df_a, df_b, on='term', suffixes=('_A', '_B'))
            
            # 두 도메인 Z-score 합산 기준 상위 100개 추출
            merged['z_sum'] = merged['z_score_A'] + merged['z_score_B']
            merged['z_diff'] = abs(merged['z_score_A'] - merged['z_score_B'])
            top_100_words = merged.sort_values(by='z_sum', ascending=False).head(100).copy()
            
            # 정보 추가
            top_100_words['pair_name'] = f"{name_a} vs {name_b}"
            top_100_words['pair_pearson'] = pearson_val
            
            all_detailed_data.append(top_100_words)

    # 3. 하나의 데이터프레임으로 합쳐서 저장
    final_df = pd.concat(all_detailed_data, ignore_index=True)
    
    # 컬럼 순서 정리
    cols = ['pair_name', 'pair_pearson', 'term', 'z_score_A', 'z_score_B', 'z_sum', 'z_diff']
    final_df = final_df[cols]
    
    final_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\n✨ 추출 완료! 파일 경로: {save_path}")
    print(f"📊 총 {len(final_df)}개의 단어 데이터가 정리되었습니다.")

if __name__ == "__main__":
    extract_detailed_words()