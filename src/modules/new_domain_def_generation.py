# src/modules/new_domain_def_generation.py

from tqdm import tqdm
import json
import os

def new_domain_def_generation(client, n=1):
    """
    [데이터 생성 파이프라인 - 1번 박스: 도메인 메타데이터 확장 자동화]
    
    사용자가 지정한 개수(n)만큼 새로운 도메인 정의를 LLM에 요청하고, 
    기존 데이터와 중복되지 않도록 관리하며 히스토리 파일에 일괄 병합합니다.
    
    Args:
        client (OpenAI): OpenAI API 호출을 위한 인증된 클라이언트 객체.
        n (int): 한 번에 생성하고자 하는 신규 도메인의 총 개수 (기본값: 1).
    """
    
    # ==========================================================
    # STEP 1. 파일 시스템 경로 및 디렉토리 환경 설정
    # ==========================================================
    # os.path.abspath(__file__)를 통해 스크립트 실행 위치에 관계없이 절대 경로를 확보합니다.
    current_script_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_script_path)
    
    # 1번 박스의 결과물들이 저장될 메타데이터 폴더와 파일 경로를 정의합니다.
    metadata_dir = os.path.join(current_dir, "new_domain_generation_metadata")
    history_file = os.path.join(metadata_dir, "domain_form_history.json")
    generated_file = os.path.join(metadata_dir, "generated_domain_form.json")

    # 메타데이터 저장용 폴더가 없다면 자동으로 생성하여 에러를 방지합니다.
    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)

    # ==========================================================
    # STEP 2. 기존 도메인 히스토리 로드 (기존 도메인 파악)
    # ==========================================================
    # 기존에 어떤 도메인들이 생성되었는지 파악하여 LLM에게 중복 제외를 요청하기 위한 로드 단계입니다.
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    else:
        # 파일이 없을 경우 파이프라인 규격에 맞는 초기 구조를 생성합니다.
        full_data = {"domain_counts": 0, "domain_forms": {}}

    # 실제 도메인 정의 데이터들이 들어있는 딕셔너리를 추출합니다.
    domain_forms = full_data.get("domain_forms", {})
    
    # 이번 실행 루프 동안 생성된 신규 도메인들을 임시로 누적할 딕셔너리입니다.
    newly_generated_total = {}

    # ==========================================================
    # STEP 3. n회 반복 생성 루프 시작 (LLM 호출 및 설계도 작성)
    # ==========================================================
    for i in tqdm(range(n), desc="New Domain Definition Generation"):
        # [중복 방지 로직] 
        # 1. 파일에 저장된 기존 도메인 타이틀 목록
        # 2. 앞선 루프(i-1번째 등)에서 방금 막 생성된 도메인 타이틀 목록
        # 위 두 가지를 모두 합쳐서 LLM에게 중복 생성을 하지 않도록 가이드합니다.
        current_existing_titles = [v['domain_title'] for v in domain_forms.values()] + \
                                  [v['domain_title'] for v in newly_generated_total.values()]
        
        # [ID 자동 채번 로직]
        # 모든 기존 Key와 현재 루프에서 생성된 Key 중 가장 큰 값을 찾아 다음 번호를 부여합니다.
        all_keys = list(domain_forms.keys()) + list(newly_generated_total.keys())
        last_id_num = max([int(k) for k in all_keys]) if all_keys else 0
        new_id = f"{last_id_num + 1:03d}"  # 예: 013 -> 014 (3자리 패딩)

        print(f"--- [{i+1}/{n}] ID: {new_id} 신규 도메인 설계도 생성 요청 중 ---")

        # 2번 박스(데이터 생성기)에서 고품질 합성 데이터를 뽑아내기 위한 핵심 프롬프트입니다.
        prompt = f"""
        [지침] 기존 생성 목록에 없는 새로운 산업 도메인 1개를 선정하여 상세 정의를 작성하세요.
        기존 생성 목록: {current_existing_titles}
        
        [출력 규칙] 반드시 다음 JSON 스키마 구조를 엄격히 준수하여 응답하세요:

        {{
            "{new_id}": {{
                "domain_title": "영문 도메인명 (Snake_case)",
                "domain_name": "한국어 공식 명칭",
                "domain_definition": "해당 도메인이 다루는 데이터와 문서의 성격에 대한 아주 상세한 설명",
                "generation_form": "주요 데이터 항목 및 문서 유형 예시 (쉼표 구분)",
                "domain_constraints": ["기술적/비즈니스적 제약 조건 리스트"],
                "required_fields": ["JSON 필수 키값 리스트"],
                "sample_data_schema": {{ "키명": "데이터타입/설명" }},
                "key_entities": ["데이터 내에 등장할 주요 역할군/개체"],
                "data_volatility": "데이터들 사이의 무작위성(Randomness) 가이드라인",
                "output_language": "한국어 기반 (전문 용어는 영어 권장)",
                "domain_system_prompt": "이 도메인의 전문가 페르소나와 데이터 생성 시 주의사항"
            }}
        }}
        """

        try:
            # 외부에서 주입된 OpenAI 클라이언트를 사용하여 응답을 요청합니다.
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": "당신은 고도로 구조화된 데이터를 설계하는 데이터 엔지니어입니다. 반드시 JSON으로만 답변하세요."},
                    {"role": "user", "content": prompt}
                ],
                # JSON Mode를 활성화하여 응답의 구조적 안정성을 확보합니다.
                response_format={ "type": "json_object" }
            )

            # 생성된 JSON 텍스트를 Python 딕셔너리로 역직렬화합니다.
            generated_content = response.choices[0].message.content
            new_entry = json.loads(generated_content)
            
            # 이번 루프의 성공 결과물을 임시 저장소에 추가합니다.
            newly_generated_total.update(new_entry)
            print(f"    -> '{list(new_entry.values())[0]['domain_name']}' 설계도 생성 완료")

        except Exception as e:
            # 네트워크 오류, API 제한 등 예외 발생 시 로그를 남기고 다음 순번으로 진행합니다.
            print(f"    [Error] {i+1}번째 도메인 생성 실패: {str(e)}")
            continue

    # ==========================================================
    # STEP 4. Result & Merge 단계 (파일 일괄 저장 및 병합)
    # ==========================================================
    # 루프를 통해 하나라도 새로운 도메인이 생성되었다면 저장을 진행합니다.
    if newly_generated_total:
        # 1. [Result] 이번 실행(n개 루프)에서 생성된 전체 결과물만 별도로 기록합니다.
        with open(generated_file, 'w', encoding='utf-8') as f:
            json.dump(newly_generated_total, f, indent=4, ensure_ascii=False)

        # 2. [Merge] 전체 히스토리 데이터에 이번 신규 도메인들을 통합합니다.
        domain_forms.update(newly_generated_total)
        
        # 최상위 카운트 정보를 현재 등록된 총 도메인 수로 갱신합니다.
        full_data["domain_counts"] = len(domain_forms)
        full_data["domain_forms"] = domain_forms

        # 최종 통합 설계도를 history_file 경로에 덮어씌워 보관합니다.
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(full_data, f, indent=4, ensure_ascii=False)
        
        print(f"\n--- [파이프라인 완료] {len(newly_generated_total)}개의 신규 도메인이 히스토리에 최종 병합되었습니다. ---")
        print(f"--- [현재 상태] 총 보유 도메인 수: {full_data['domain_counts']}개 ---")
    else:
        # 모든 루프가 실패했거나 n이 0인 경우 실행됩니다.
        print("\n[System] 새로 생성된 도메인이 없어 병합 및 저장을 수행하지 않았습니다.")