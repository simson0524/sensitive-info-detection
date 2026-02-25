# src/processes/process_4.py

import torch
import os
import logging
from datetime import datetime
from torch.utils.data import DataLoader

# Modules
from src.modules.ner_evaluator import Evaluator
from src.models.ner_roberta import RobertaNerModel
# TODO!!!

# Database
from src.database.connection import db_manager
from src.database import crud

# Utils
from src.utils.common import ensure_dir, save_logs_to_csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math

# test
import difflib

def run_process_4(config: dict, context: dict):
    """
    [Process 4] 모델 보완 추론 및 Hybrid 검증 프로세스
    
    1. 규칙(사전/Regex)이 찾은 결과를 DB에서 로드하여 메모리에 매핑.
    2. 학습된 Best Model로 전체 검증 데이터에 대해 추론 수행.
    3. 모델 결과와 규칙 결과를 비교하여 유형 분류 및 통계 산출:
       - Double Check: 규칙도 찾고 모델도 찾음 (신뢰도 높음)
       - Model Complement: 규칙은 못 찾았는데 모델이 찾음 (모델의 기여도)
       - Rule Only: 규칙은 찾았는데 모델은 못 찾음 (모델의 한계)
    4. 분석 결과 및 로그 DB 저장 & CSV 추출.
    """
    
    # ==============================================================================
    # [Step 1] 설정 및 로거 초기화
    # ==============================================================================
    experiment_code = context['experiment_code']
    device = context['device']
    preprocessor = context['preprocessor']
    
    path_conf = config['path']
    train_conf = config['train']

    logger = logging.getLogger(experiment_code)
    logger.info(f"🚀 [Process 4] Start Hybrid Inference & Analysis")

    # ==============================================================================
    # [Step 2] 규칙 기반 탐지 결과 로드 (Process 2 & 3)
    # ==============================================================================
    logger.info("Loading Rule-based detection results from DB...")
    
    rule_hits = {}
    
    with db_manager.get_db() as session:
        for proc_code in ["process_2", "process_3"]:
            logs = crud.get_inference_sentences(session, experiment_code, proc_code, 1)
            for log in logs:
                sid = log['sentence_id']
                if sid not in rule_hits:
                    rule_hits[sid] = {}
                
                res = log.get('sentence_inference_result', {})
                results_list = res.get('inference_results', [])
                
                for r in results_list:
                    if r.get('match_result') in ['hit', 'prediction']:
                        word = r['word']
                        label = r['label']
                        rule_hits[sid][word] = label

    logger.info(f"Loaded rule hits for {len(rule_hits)} sentences.")

    # ==============================================================================
    # [Step 3] Best Model 로드
    # ==============================================================================
    logger.info("Loading Best Model from Checkpoint...")
    
    encoder = context['model'].encoder 
    num_labels = len(preprocessor.ner_label2id)
    
    best_model = RobertaNerModel(
        encoder=encoder,
        num_classes=num_labels,
        use_focal=False 
    ).to(device)
    
    ckpt_path = os.path.join(
        path_conf['checkpoint_dir'], experiment_code, f"{experiment_code}_epoch_{context['best_epoch']}.pt"
    )
    
    if os.path.exists(ckpt_path):
        best_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        logger.info(f"✅ Loaded weights from {ckpt_path}")
    else:
        logger.warning(f"⚠️ Checkpoint not found at {ckpt_path}. Using current model state.")
        best_model = context['model']

    # ==============================================================================
    # [Step 4] 추론 및 비교 분석 (Hybrid Logic)
    # ==============================================================================
    evaluator = Evaluator(
        best_model, 
        device, 
        preprocessor.tokenizer, 
        preprocessor.ner_id2label 
    )

    result = evaluator.evaluate(context['test_loader']) # 이거 mode="test"를 없애야 할 것 같은데...? ㅇㅇ 없앰 ㅋ
    raw_logs = result['logs']

    stats = {
        "double_check": 0, "model_complement": 0, "rule_only": 0, "total_model_detected": 0
    }

    processed_logs = []

    for log in raw_logs:
        sid = log['sentence_id']
        model_results = log['sentence_inference_result']['inference_results']

        rule_findings = rule_hits.get(sid, {}).copy() 
        
        # 1. 모델 탐지 결과 순회
        for entity in model_results:
            word = entity['word']
            if word in rule_findings:
                entity['hybrid_status'] = "Double Check"
                stats['double_check'] += 1
                rule_findings.pop(word, None)
            else:
                entity['hybrid_status'] = "Model Complement"
                stats['model_complement'] += 1
            stats['total_model_detected'] += 1
            
        # 2. Rule Only 계산
        for r_word, r_label in rule_findings.items():
            stats['rule_only'] += 1
            model_results.append({
                "word": r_word,
                "label": r_label,
                "start": -1, 
                "end": -1,
                "hybrid_status": "Rule Only (Model Missed)"
            })
        
        log['sentence_inference_result']['inference_results'] = model_results
        log['sentence_inference_result']['entity_count'] = len(model_results)
        processed_logs.append(log)

    # 비율 계산
    total_detections = stats['double_check'] + stats['model_complement'] + stats['rule_only']
    if total_detections > 0:
        stats['ratio_double_check'] = round(stats['double_check'] / total_detections, 4)
        stats['ratio_complement'] = round(stats['model_complement'] / total_detections, 4)
        stats['ratio_rule_only'] = round(stats['rule_only'] / total_detections, 4)
    else:
        stats.update({'ratio_double_check': 0, 'ratio_complement': 0, 'ratio_rule_only': 0})

    logger.info(f"📊 Hybrid Analysis Result: {stats}")

    # ==============================================================================
    # [Step 5] DB 저장 및 CSV 추출    ### 여기 레이어 좀 무너져있음;;; ㅠ 시바
    # ==============================================================================
    
    # CSV 저장 경로 생성
    log_save_dir = os.path.join(path_conf['log_dir'], experiment_code)
    ensure_dir(log_save_dir)

    with db_manager.get_db() as session:
        # 5-1. 결과 요약 저장
        crud.create_process_result(session, {
            "experiment_code": experiment_code,
            "process_code": "process_4", 
            "process_epoch": 1,
            "process_start_time": datetime.now(), 
            "process_end_time": result.get('end_time', datetime.now()),
            "process_duration": result['metrics'].get('duration', 0.0),
            "process_results": {
                "hybrid_stats": stats,
                "base_metrics": result['metrics']
            }
        })

        # 5-2. 문장 로그 저장 (Bulk Insert)
        # FK 주입
        for log in processed_logs:
            log['experiment_code'] = experiment_code
            log['process_code'] = "process_4"
            log['process_epoch'] = 1
        
        crud.bulk_insert_inference_sentences(session, processed_logs)
        logger.info(f"Saved {len(processed_logs)} hybrid inference logs to DB.")
        
        # 5-3. 각 GT라벨에 대하여 Pred된 라벨들(with conf score)에 대한 plot과 정탐 미탐 오탐 개수와 비율을 나타낸 텍스트를 하나의 Plot에 추가
        # key: GT label, value: PRED label, 
        final_results = {}
        for log in raw_logs:
            sentence_inference_result_list = log['sentence_inference_result']
            inferenced_results_list = sentence_inference_result_list['inference_results']
            token_comparison_list = sentence_inference_result_list['token_comparison']

            # [수정] 딕셔너리 객체 대신 고유 키를 사용하여 매핑 딕셔너리 생성
            non_normal_token_comparison = {}
            for token_comparison in token_comparison_list:
                pred_entity = token_comparison['pred_entity']
                # 단어와 시작/끝 위치를 조합하여 고유 키 생성
                unique_key = f"{pred_entity['word']}_{pred_entity['start']}_{pred_entity['end']}"
                non_normal_token_comparison[unique_key] = token_comparison

            # 확인(일반정보)
            for token_comparison in token_comparison_list:
                if token_comparison['pred_label'] == "일반정보":
                    pred_result = token_comparison['pred_entity']
                    pred_result['score'] = 0.0
                    pred_result['sentence_id'] = sentence_inference_result_list['sentence_id']
                    pred_result['origin_sentence'] = sentence_inference_result_list['origin_sentence']
                    pred_result['gt_word'] = token_comparison['gt_entity']['word']

                    gt_label = token_comparison['gt_label']
                    if gt_label in final_results:
                        final_results[gt_label].append(pred_result)
                    else:
                        final_results[gt_label] = [pred_result]

            # 확인(일반정보 제외)
            for inferenced_result in inferenced_results_list:
                # [수정] 조회할 때도 동일한 규칙으로 키 생성
                current_key = f"{inferenced_result['word']}_{inferenced_result['start']}_{inferenced_result['end']}"
                
                # 생성한 문자열 키로 존재 여부 확인 (더 이상 에러 발생 X)
                if current_key in non_normal_token_comparison:
                    curr_comparison = non_normal_token_comparison[current_key]
                    inferenced_result['gt_word'] = curr_comparison['gt_entity']['word']
                    inferenced_result['score'] = curr_comparison['score']
                    inferenced_result['sentence_id'] = sentence_inference_result_list['sentence_id']
                    inferenced_result['origin_sentence'] = sentence_inference_result_list['origin_sentence']

                    gt_label = curr_comparison['gt_label']
                    if gt_label in final_results:
                        final_results[gt_label].append(inferenced_result)
                    else:
                        final_results[gt_label] = [inferenced_result]
                
                else:
                    # GT가 '일반정보'인 경우
                    inferenced_result['score'] = 0.0
                    inferenced_result['sentence_id'] = sentence_inference_result_list['sentence_id']
                    inferenced_result['origin_sentence'] = sentence_inference_result_list['origin_sentence']
                    inferenced_result['gt_word'] = 'NULL'

                    if "일반정보" in final_results:
                        final_results["일반정보"].append(inferenced_result)
                    else:
                        final_results["일반정보"] = [inferenced_result]

        # --- 레이어 전혀 안지키이이임 --- #
        # --- 999번 도메인 z-score 불러오기 ---
        z_score = None

        with db_manager.get_db() as session:
            z_score = list(crud.get_dtm_by_domain(session, 999))

        # print('[debug] z_score\n\n', z_score)

        z_score_by_term = {}

        for data in z_score:
            if data['term'] in z_score_by_term and data['z_score'] < z_score_by_term[data['term']]:
                continue
            z_score_by_term[data['term']] = data['z_score']

        for gt_label, result_dict in final_results.items():
            for result in result_dict:
                gt_word = result['gt_word']
                pred_word = result['word']
                answer_bytes = difflib.SequenceMatcher(None, gt_word, pred_word)
                result['ratio_score'] = answer_bytes.ratio()
                result['z_score'] = z_score_by_term.get(gt_word, 0.0)
    
        for gt_label, results_list in final_results.items():
            # 1. '일반정보' 건너뛰기
            if gt_label == '일반정보':
                continue
            
            # 해당 gt_label의 전체 샘플 수 계산
            total_samples = len(results_list)
            if total_samples == 0:
                continue

            # 2. 결과 데이터를 pred_label 별로 그룹화 (Subplot 생성을 위해)
            from collections import defaultdict
            grouped_data = defaultdict(list)
            for res in results_list:
                grouped_data[res['label']].append(res)

            num_subs = len(grouped_data)
            fig, axes = plt.subplots(num_subs, 1, figsize=(10, 2 * num_subs), squeeze=False)
            
            for idx, (pred_label, samples) in enumerate(grouped_data.items()):
                ax = axes[idx, 0]
                
                # 데이터 추출
                z_scores = [min(s.get('z_score', 0.0), 4.0) for s in samples]
                ratio_scores = [s.get('ratio_score', 0.0) for s in samples]

                # 이상치 수 확인
                outlier = sum(1 for s in samples if s.get('z_score', 0.0) > 4.0)

                # boxplot 기준
                z_scores_arr = np.array(z_scores)
                p100, p75, p50, p25, p0 = np.percentile(z_scores_arr, [100, 75, 50, 25, 0])
                
                # 통계치 계산
                count = len(samples)
                percentage = (count / total_samples) * 100
                z_avg = np.mean(z_scores)
                z_std = np.std(z_scores)
                
                # 3. 산점도 그리기
                ax.scatter(z_scores, ratio_scores, alpha=0.6, edgecolors='w', label=f'Samples (n={count})')
                
                # 3. 각 지점에 수직 점선 그리기 (axvline)
                # 중앙값 (50%) - 가장 중요하므로 빨간색 실선 혹은 굵은 점선
                ax.axvline(p50, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Median ({p50:.2f})')
                
                # 25% 및 75% 지점 - 오렌지색 점선
                ax.axvline(p25, color='orange', linestyle=':', alpha=0.6, label=f'Q1/Q3 ({p25:.2f}, {p75:.2f})')
                ax.axvline(p75, color='orange', linestyle=':', alpha=0.6)
                
                # 0% 및 100% 지점 (최솟값/최댓값) - 회색 아주 연한 점선
                ax.axvline(p0, color='green', linestyle='-.', alpha=0.4, label='Min/Max')
                ax.axvline(p100, color='green', linestyle='-.', alpha=0.4)

                # 4. 범례 추가 (선의 의미를 알기 위해)
                ax.legend(loc='upper right', fontsize='small')

                # 5. (선택사항) x축 범위 최적화
                # 4.0으로 clipping 하셨다면 범위를 0~4.5 정도로 잡아주면 깔끔합니다.
                ax.set_xlim(min(0, p0 - 0.5), max(4, p100 + 1.5))
                
                # # 4. 통계 정보 텍스트 박스 추가
                # stats_text = (f"Count: {count} ({percentage:.1f}%)\n"
                #               f"Outlier Count: {outlier}"
                #               f"Z-Score Mean/Std: {z_avg:.3f}/{z_std:.3f}\n")
                
                # # 그래프 내 우측 상단에 박스 배치
                # ax.text(0.05, 0.5, stats_text, transform=ax.transAxes, fontsize=10,
                #         verticalalignment='top', horizontalalignment='right',
                #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # 설정
                ax.set_title(f"Exeriment: {experiment_code} | GT: {gt_label} | PRED: {pred_label}\nCount: {count} ({percentage:.1f}%) | Outlier Count: {outlier}", fontsize=14, fontweight='bold')
                ax.set_xlabel("Z-Score (x-axis)")
                ax.set_ylabel("Ratio Score (y-axis)")
                ax.grid(True, linestyle=':', alpha=0.7)
                ax.set_ylim(-0.1, 1.1) # Ratio Score는 0~1 사이이므로

            plt.tight_layout()
            
            # 5. 저장
            png_file_path = os.path.join(log_save_dir, f"{gt_label}_inference_results.png")
            plt.savefig(png_file_path)
            plt.close(fig)
            print(f"📊 시각화 완료: {png_file_path}")

        # # --- 시각화 섹션 ---
        # sns.set_theme(style="whitegrid")
        # plt.rcParams['font.family'] = 'NanumGothic' 
        # plt.rcParams['axes.unicode_minus'] = False

        # gt_labels = list(final_results.keys())
        # n_labels = len(gt_labels)
        # n_cols = 2
        # n_rows = math.ceil(n_labels / n_cols)

        # fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
        # if n_labels == 1: axes = [axes] # 라벨이 1개일 경우 대비
        # else: axes = axes.flatten()

        # for idx, gt_label in enumerate(gt_labels):
        #     ax = axes[idx]
        #     df = pd.DataFrame(final_results[gt_label])
            
        #     # X축 정렬 및 개수(n=) 포함 라벨 생성
        #     sorted_labels = sorted(df['label'].unique())
        #     label_counts = df['label'].value_counts()
        #     xtick_with_counts = [f"{l}\n(n={label_counts[l]})" for l in sorted_labels]
            
        #     # 통계 계산
        #     total_count = len(df)
        #     match_count = len(df[df['label'] == gt_label])
        #     mismatch_count = total_count - match_count
        #     match_ratio = (match_count / total_count * 100) if total_count > 0 else 0
            
        #     # Blending Plot (Strip + Box)
        #     sns.stripplot(data=df, x='label', y='score', order=sorted_labels,
        #                   ax=ax, jitter=0.2, size=4, alpha=0.5, palette="magma")
        #     sns.boxplot(data=df, x='label', y='score', order=sorted_labels,
        #                 ax=ax, whis=np.inf, color="0.9", width=0.4, boxprops=dict(alpha=0.3))

        #     # 제목 및 축 설정
        #     title_str = (f"GT: {gt_label}\n"
        #                 f"MATCH: {match_count} ({match_ratio:.1f}%) | "
        #                 f"MISMATCH: {mismatch_count}")
            
        #     ax.set_title(title_str, fontsize=14, fontweight='bold', pad=15)
        #     ax.set_xticks(range(len(sorted_labels)))
        #     ax.set_xticklabels(xtick_with_counts, rotation=30, ha='right')
        #     ax.set_ylim(-0.05, 1.05)
        #     ax.set_xlabel("Predicted Labels (Count)", fontsize=10)
        #     ax.set_ylabel("Confidence Score", fontsize=10)

        # # 빈 서브플롯 제거
        # for j in range(idx + 1, len(axes)):
        #     fig.delaxes(axes[j])

        # plt.tight_layout()
        # png_file_path = os.path.join(log_save_dir, "total_inference_results.png")
        # plt.savefig(png_file_path, dpi=150)
        # logger.info(f"Saved Plot to {png_file_path}")

        # 5-4. CSV 파일 추출
        csv_file_name = f"{experiment_code}_process_4_1_inference_sentences.csv"
        csv_file_path = os.path.join(log_save_dir, csv_file_name)
        
        all_data_for_csv = []
        for gt_label, records in final_results.items():
            for record in records:
                row = record.copy()
                row['gt_label'] = gt_label
                all_data_for_csv.append(row)

        df_final = pd.DataFrame(all_data_for_csv)
        if not df_final.empty:
            cols = ['gt_label'] + [c for c in df_final.columns if c != 'gt_label']
            df_final = df_final[cols]
            df_final.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
            logger.info(f"Saved CSV log to {csv_file_path}")

    logger.info("[Process 4] Completed.")
    return context