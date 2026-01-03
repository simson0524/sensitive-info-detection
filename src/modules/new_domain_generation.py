# scr/modules/new_domain_generation.py

import os
import json
import random
import shutil
import zipfile
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.modules.new_domain_annotation import new_domain_annotation

def new_domain_generation(
    client, 
    target_count=100, 
    metadata_rel_path='new_domain_generation_metadata/generated_domain_form.json', 
    name_pool_rel_path='new_domain_generation_metadata/name_pool.json',
    label_desc_rel_path='new_domain_generation_metadata/label_description.json',
    format_sample_rel_path='new_domain_generation_metadata/document_format.json'
):
    """
    [데이터 생성 파이프라인 - 2번 박스: 도메인별 데이터 병렬 생성 및 실시간 모니터링]
    """
    
    # ---------------------------------------------------------
    # STEP 1. 경로 설정 및 리소스 로드
    # ---------------------------------------------------------
    module_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(module_dir, "../../"))
    
    metadata_path      = os.path.join(module_dir, metadata_rel_path)
    name_pool_path     = os.path.join(module_dir, name_pool_rel_path)
    label_desc_path    = os.path.join(module_dir, label_desc_rel_path)
    format_sample_path = os.path.join(module_dir, format_sample_rel_path)
    
    zip_storage_dir = os.path.join(project_root, "data", "zip_raw_data")
    os.makedirs(zip_storage_dir, exist_ok=True)

    with open(metadata_path, 'r', encoding='utf-8') as f:
        if metadata_rel_path=='new_domain_generation_metadata/generated_domain_form.json':
            new_domains_data = json.load(f)
        else:
            new_domains_history_data = json.load(f)
            new_domains_data = new_domains_history_data["domain_forms"]

    with open(name_pool_path, 'r', encoding='utf-8') as f:
        name_pool = json.load(f)
    with open(label_desc_path, 'r', encoding='utf-8') as f:
        label_descriptions = json.load(f)
    with open(format_sample_path, 'r', encoding='utf-8') as f:
        document_format_sample = json.load(f)

    # ---------------------------------------------------------
    # STEP 2. 도메인 단위 병렬 프로세스 제어
    # ---------------------------------------------------------
    domain_ids = list(new_domains_data.keys())
    
    # [Main tqdm] 전체 도메인 완료 상태 표시 바 (최상단)
    main_pbar = tqdm(total=len(domain_ids), desc="Total Domains", position=0, leave=True)

    # 병렬 처리를 위해 ThreadPoolExecutor 사용
    # max_workers를 통해 한 번에 화면에 보일 서브 바의 개수를 조절할 수 있습니다.
    with ThreadPoolExecutor(max_workers=min(len(domain_ids), 5)) as executor:
        futures = {
            executor.submit(
                _process_domain_pipeline, 
                client, 
                domain_id, 
                new_domains_data[domain_id], 
                target_count, 
                name_pool, 
                label_descriptions, 
                document_format_sample,
                zip_storage_dir, 
                project_root,
                idx + 1 # 개별 tqdm의 출력 라인 위치 지정
            ): domain_id for idx, domain_id in enumerate(domain_ids)
        }

        for future in as_completed(futures):
            d_id = futures[future]
            try:
                # 1. 2번 박스 완료 후 temp_dir를 받음
                temp_working_dir = future.result() 
                d_title = new_domains_data[d_id]['domain_title']
                
                # 2. 여기서 바로 Annotation + Zip 실행
                if temp_working_dir:
                    new_domain_annotation(temp_working_dir, d_id, d_title, zip_storage_dir)
            except Exception as e:
                # 오류 메시지가 진행 바를 가리지 않도록 tqdm.write 사용
                tqdm.write(f"\n[Error] 도메인 {d_id} 처리 실패: {e}")
            finally:
                main_pbar.update(1)

    main_pbar.close()
    print(f"\n[System] 모든 작업 완료. 결과물 위치: {zip_storage_dir}")


def _process_domain_pipeline(client, domain_id, domain_info, target_count, name_pool, label_desc, format_sample, zip_storage_dir, project_root, position):
    """
    [내부 함수] 각 도메인 스레드 내에서 문서를 생성하고, 
    label_desc를 참조하여 정답지(extracted_answers)를 동시에 추출합니다.
    """
    domain_title = domain_info['domain_title']
    domain_name = domain_info['domain_name']
    
    temp_dir = os.path.join(project_root, "data", f"temp_{domain_id}")
    os.makedirs(temp_dir, exist_ok=True)

    success_count = 0
    error_count = 0
    fail_streak = 0
    
    # 해당 도메인의 모든 문장에서 추출된 정답을 모으는 리스트
    total_answer_sheet = []

    sub_pbar = tqdm(
        total=target_count, 
        desc=f"[{domain_id}] {success_count}/{target_count} (Error : {error_count}회)", 
        position=position, 
        leave=False 
    )

    while success_count < target_count:
        if fail_streak > 20: 
            tqdm.write(f"\n[Warning] {domain_id} pipeline aborted.")
            break

        doc_idx = success_count + 1
        doc_id_str = f"{doc_idx:04d}"
        full_doc_id = f"{domain_id}_{doc_id_str}"
        sampled_names = random.sample(name_pool, k=min(12, len(name_pool)))

        # 보강된 통합 메인 프롬프트
        main_prompt = f"""
        # Task
        1. Create a highly realistic, professional synthetic document for the '{domain_name}' domain.
        2. Strictly identify and extract sensitive information based on 'Label Descriptions'.

        # Expert Role & Persona
        {domain_info['domain_system_prompt']}

        # Domain Context & Quality Requirements
        - Definition: {domain_info['domain_definition']}
        - Required Content: {domain_info['generation_form']}
        - Specific Constraints: {domain_info['domain_constraints']}
        - Output Language: MUST be "Korean" (Technical terms allowed in English).

        # Document Length & Structure Rules (CRITICAL)
        - Sentence Length: Each sentence MUST be under 300 characters for readability.
        - Document Volume: Generate at least 25 sentences to ensure the document feels like a real-world file (not a summary).
        - Structure: Organize the content logically (e.g., Introduction -> Detailed Body -> Conclusion/Action Items).
        - Richness: Use professional terminology and detailed descriptions relevant to {domain_name}.

        # Label Descriptions (Reference for Answer Extraction)
        {json.dumps(label_desc, indent=2, ensure_ascii=False)}

        # Naming Rules
        Use names from this list for person entities: {", ".join(sampled_names)}

        # Output Structure (STRICT JSON)
        Return a single JSON object with the following two keys:

        1. "document_data": 
           Follow this structure: {json.dumps(format_sample, indent=2, ensure_ascii=False)}
           - document_id: "{full_doc_id}"
           - document_title: A realistic Korean title.
           - data: List of sentence objects.
           - annotations: MUST be an empty list [] for all sentences.
           - sentence ID format: "sample_{domain_id}_{doc_id_str}_00000X" (sequence)

        2. "extracted_answers":
           A list of sensitive words found in the "document_data" based on the 'Label Descriptions'.
           Each object must contain:
           - "word": The extracted sensitive text.
           - "sentence_id": The ID of the sentence where the word was found.
           - "label": The specific label (개인정보 / 준식별자 / 기밀정보).

        # Rules
        - Ensure logical consistency between the document content and the extracted answers.
        - Do not omit any sensitive information defined in the 'Label Descriptions'.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": main_prompt}],
                response_format={ "type": "json_object" }
            )
            result = json.loads(response.choices[0].message.content)
            
            # 1. 문서 데이터(JSON) 처리
            doc_json = result["document_data"]
            with open(os.path.join(temp_dir, f"{full_doc_id}.json"), 'w', encoding='utf-8') as f:
                json.dump(doc_json, f, indent=4, ensure_ascii=False)
            
            # 2. 정답지 데이터(CSV용) 누적
            extracted = result.get("extracted_answers", [])
            for ans in extracted:
                total_answer_sheet.append({
                    "word": ans["word"],
                    "document": full_doc_id,
                    "sentence": ans["sentence_id"],
                    "label": ans["label"]
                })

            success_count += 1
            fail_streak = 0
            sub_pbar.update(1)
            sub_pbar.set_description(f"[{domain_id}] {success_count}/{target_count} (Error : {error_count}회)")

        except Exception:
            error_count += 1
            fail_streak += 1
            sub_pbar.set_description(f"[{domain_id}] {success_count}/{target_count} (Error : {error_count}회)")
            continue

    sub_pbar.close()

    # answer_sheet.csv 저장 (도메인이 한 장도 생성되지 않았을 경우 대비)
    if success_count > 0 and total_answer_sheet:
        csv_path = os.path.join(temp_dir, "answer_sheet.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=["word", "document", "sentence", "label"])
            writer.writeheader()
            writer.writerows(total_answer_sheet)
        return temp_dir # 성공적으로 생성된 경우에만 경로 리턴
    else:
        # 실패한 경우 임시 폴더 삭제 후 None 리턴
        shutil.rmtree(temp_dir)
        return None