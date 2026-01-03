# scripts/run_new_domain_generation.py

import os
import sys
import yaml
from openai import OpenAI

# 프로젝트 루트 경로 설정
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(PROJECT_ROOT)

# 최상위 함수들만 임포트
from src.modules.new_domain_def_generation import new_domain_def_generation
from src.modules.new_domain_generation import new_domain_generation

def main():
    # 1. 설정 로드
    config_path = os.path.join(PROJECT_ROOT, "configs/new_domain_generation_config.yaml")
    cfg = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))
    gen_cfg = cfg['generation_settings']
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # ---------------------------------------------------------
    # [STEP 1] 1번 박스: 도메인 정의 생성 (Mode가 True일 때만)
    # ---------------------------------------------------------
    if gen_cfg['mode']:
        print(f"[System] Mode: New Domain Definition Generation (n={gen_cfg['new_domain_n']})")
        new_domain_def_generation(client, n=gen_cfg['new_domain_n'])
        metadata_rel_path = cfg['paths']['new_metadata_rel_path']
    else:
        print(f"[System] Mode: History-based Data Generation")
        metadata_rel_path = cfg['paths']['history_metadata_rel_path']

    # ---------------------------------------------------------
    # [STEP 2] 2번 박스(+내부 3번 박스): 데이터 생성 및 어노테이션 패키징
    # ---------------------------------------------------------
    # 사용자님이 정성껏 만든 new_domain_generation 함수 하나로 모든 생성을 끝냅니다.
    if gen_cfg['mode']:
        new_domain_generation(
            client=client,
            target_count=gen_cfg['target_count']
        )
    else:
        new_domain_generation(
            client=client,
            target_count=gen_cfg['target_count'],
            metadata_rel_path=metadata_rel_path
        )

    print("\n[Final System] 모든 파이프라인 프로세스가 성공적으로 완료되었습니다.")

if __name__ == "__main__":
    main()