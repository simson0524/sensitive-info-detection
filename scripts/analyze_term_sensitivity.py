import os
import pandas as pd
import numpy as np
from src.database.connection import db_manager
from src.database.models import DomainTermMatrix
from tqdm import tqdm

def analyze_term_sensitivity_v2():
    output_dir = "outputs/logs"
    os.makedirs(output_dir, exist_ok=True)
    
    mixed_csv = os.path.join(output_dir, "term_stats_mixed_sensitivity_detailed.csv")
    uniform_csv = os.path.join(output_dir, "term_stats_uniform_sensitivity_detailed.csv")

    print("📦 DB에서 전체 데이터를 로드 중입니다...")
    with db_manager.get_db() as session:
        # domain_id를 포함하여 쿼리
        query = session.query(
            DomainTermMatrix.term, 
            DomainTermMatrix.domain_id,
            DomainTermMatrix.z_score, 
            DomainTermMatrix.is_sensitive_label
        )
        df = pd.read_sql(query.statement, session.bind)

    if df.empty:
        print("❌ 데이터가 없습니다.")
        return

    print(f"📊 총 {len(df)}행 분석 및 데이터 그룹화 중...")

    # 1. 단어별 상세 정보 생성을 위한 함수 정의
    def aggregate_details(group):
        # z_score와 label을 보기 좋게 리스트 문자열로 변환
        # 예: "ID_1: 2.5(T), ID_2: 1.2(F)"
        details = [
            f"ID_{row.domain_id}: {row.z_score:.2f}({'T' if row.is_sensitive_label else 'F'})"
            for row in group.itertuples()
        ]
        
        return pd.Series({
            'z_score_mean': group['z_score'].mean(),
            'z_score_std': group['z_score'].std(),
            'appearance_count': len(group),
            'label_count': group['is_sensitive_label'].nunique(),
            'is_sensitive_uniform': group['is_sensitive_label'].iloc[0] if group['is_sensitive_label'].nunique() == 1 else None,
            'raw_values_list': " | ".join(details) # 실제 값들을 문자열로 결합
        })

    # 2. 단어(term)별 그룹화 연산
    tqdm.pandas(desc="단어별 통계 계산 중")
    stats = df.groupby('term').progress_apply(aggregate_details).reset_index()

    # NaN 표준편차 처리
    stats['z_score_std'] = stats['z_score_std'].fillna(0)

    # 3. 케이스 분류
    df_mixed = stats[stats['label_count'] > 1].copy()
    df_uniform = stats[stats['label_count'] == 1].copy()

    # 4. 정렬 (평균 Z-score 내림차순)
    df_mixed = df_mixed.sort_values(by='z_score_mean', ascending=False)
    df_uniform = df_uniform.sort_values(by='z_score_mean', ascending=False)

    # 5. CSV 저장
    df_mixed.to_csv(mixed_csv, index=False, encoding='utf-8-sig')
    df_uniform.to_csv(uniform_csv, index=False, encoding='utf-8-sig')

    print("\n" + "="*60)
    print("✨ 상세 분석 완료! (실제 값 리스트 포함)")
    print(f"1. Mixed (라벨 혼재)  : {len(df_mixed):>6} 단어")
    print(f"2. Uniform (라벨 통일): {len(df_uniform):>6} 단어")
    print(f"📁 저장 위치: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    analyze_term_sensitivity_v2()