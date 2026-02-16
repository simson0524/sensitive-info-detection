# temp

import os
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
from src.database.connection import db_manager
from src.database.models import DomainTermMatrix, Domain
from datetime import datetime
from tqdm import tqdm

def analyze_domain_correlations():
    # 1. 경로 설정
    output_dir = "outputs/logs"
    meta_path = "/home/student1/sensitive-info-detection/src/modules/new_domain_generation_metadata/domain_form_history.json"
    os.makedirs(output_dir, exist_ok=True)
    
    summary_file = os.path.join(output_dir, "correlation_summary_report.txt")
    detail_csv = os.path.join(output_dir, "correlation_details.csv")

    # 2. JSON 메타데이터 로드
    print("📖 메타데이터(JSON)를 읽어오는 중...")
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
        
        domain_meta = {}
        for d_id_str, info in meta_data.get("domain_form", {}).items():
            d_id = int(d_id_str)
            if d_id == 999: continue  # 메타데이터에서도 999번 제외
            
            formatted_id = f"ID:{d_id_str.zfill(3)}" 
            domain_meta[d_id] = {
                "id_label": formatted_id,
                "kor_name": info.get("domain_name", "알수없음"),
                "eng_name": info.get("domain_title", "Unknown")
            }
    except Exception as e:
        print(f"❌ 메타데이터 로드 실패: {e}")
        return

    # 3. DB 데이터 로드 (999번 제외 쿼리)
    print("📦 DB에서 Z-score 데이터를 로드 중입니다 (999번 제외)...")
    with db_manager.get_db() as session:
        # 도메인 목록 가져올 때 999 제외
        domain_ids = [d[0] for d in session.query(Domain.domain_id).filter(Domain.domain_id != 999).all()]
        domain_ids = sorted(domain_ids)
        
        # 행렬 데이터 가져올 때 999 제외
        query = session.query(
            DomainTermMatrix.domain_id, 
            DomainTermMatrix.term, 
            DomainTermMatrix.z_score
        ).filter(DomainTermMatrix.domain_id != 999)
        df_all = pd.read_sql(query.statement, session.bind)

    # 4. 변수 초기화
    n = len(domain_ids)
    methods = ['pearson', 'spearman', 'kendall']
    corr_matrices = {m: pd.DataFrame(index=domain_ids, columns=domain_ids, dtype=float) for m in methods}
    for m in methods: np.fill_diagonal(corr_matrices[m].values, 1.0)

    detail_records = []
    pairs = [(domain_ids[i], domain_ids[j]) for i in range(n) for j in range(i + 1, n)]

    # 5. 상관관계 연산
    for id_a, id_b in tqdm(pairs, desc="상관관계 연산 중"):
        meta_a = domain_meta.get(id_a, {"kor_name": f"도메인{id_a}", "id_label": f"ID:{id_a}"})
        meta_b = domain_meta.get(id_b, {"kor_name": f"도메인{id_b}", "id_label": f"ID:{id_b}"})
        
        data_a = df_all[df_all['domain_id'] == id_a][['term', 'z_score']]
        data_b = df_all[df_all['domain_id'] == id_b][['term', 'z_score']]
        
        merged = pd.merge(data_a, data_b, on='term', suffixes=('_a', '_b'))
        common_count = len(merged)

        res = {
            "id_a": id_a, "id_b": id_b,
            "domain_a_kor": meta_a['kor_name'], "domain_b_kor": meta_b['kor_name'],
            "common_terms": common_count,
            "pearson": np.nan, "spearman": np.nan, "kendall": np.nan
        }

        if common_count > 2:
            va, vb = merged['z_score_a'], merged['z_score_b']
            res["pearson"], _ = pearsonr(va, vb)
            res["spearman"], _ = spearmanr(va, vb)
            res["kendall"], _ = kendalltau(va, vb)
            for m in methods:
                corr_matrices[m].at[id_a, id_b] = corr_matrices[m].at[id_b, id_a] = res[m]

        detail_records.append(res)

    # 6. 상세 결과 CSV 저장
    pd.DataFrame(detail_records).to_csv(detail_csv, index=False, encoding='utf-8-sig')

    # 7. 교수님 보고용 요약 TXT 작성
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("🔍 도메인별 Z-score 상관관계 분석 보고서 (제외 도메인: 999)\n")
        f.write(f"분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"최종 분석 도메인 수: {n}개\n")
        f.write("-" * 60 + "\n\n")

        df_detail = pd.DataFrame(detail_records)
        for m in methods:
            avg_val = df_detail[m].dropna().mean()
            f.write(f"📊 {m.capitalize()} 평균 상관계수: {avg_val:.4f}\n")

        f.write("\n[TOP 10 유사 도메인 쌍 (Pearson 기준)]\n")
        top_10 = df_detail.sort_values(by="pearson", ascending=False).head(10)
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            f.write(f"{i}. {row['domain_a_kor']} - {row['domain_b_kor']}: {row['pearson']:.4f} (공통단어 {row['common_terms']}개)\n")

    # 8. 히트맵 생성 (ID:XX 형식)
    print("🎨 히트맵 생성 중...")
    for m in methods:
        plt.figure(figsize=(24, 20))
        plot_df = corr_matrices[m].copy()
        
        labels = [domain_meta.get(idx, {"id_label": str(idx)})["id_label"] for idx in plot_df.index]
        plot_df.index = labels
        plot_df.columns = labels
        
        sns.heatmap(plot_df.astype(float), annot=False, cmap='RdBu_r', center=0)
        plt.title(f"Domain Correlation Matrix ({m.capitalize()}) - Excl. 999", fontsize=22)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"heatmap_{m}.png"))
        plt.close()

    print(f"✨ 분석 완료! 999번 도메인을 제외하고 총 {n}개 도메인에 대한 결과가 생성되었습니다.")

if __name__ == "__main__":
    analyze_domain_correlations()