import subprocess


scripts = [
    "01_preprocess_v2.py",          # 9~10차 원자료에서 11개 변수 + 결과변수(MMSE) 추출 후 병합하여 분석용 데이터셋 생성
    "02_modeling_v2.py",          # 전처리된 데이터를 바탕으로 기본 모델링 수행 및 주요 표(표3, 4, 5) 생성
    "03_robustness_check_v2.py",  # 문화여가활동 변수를 연속형에서 이분형으로 변경하여 핵심 결과의 동일성 검증
    "05_hyperparameter_tuning_v2.py", # 3.3절 초기 하이퍼파라미터가 실제 최적에 가까운지 검증 및 튜닝
    "06_weighted_analysis_v2.py", # KLoSA 9차 조사가중치를 반영하여 무가중 분석 결과와 비교하는 강건성 검정
    "07_cutoff_sensitivity_v2.py",# 인지기능저하 판정 기준(23점 이하)을 20~25점으로 변경하며 순위 및 유의성 민감도 분석
    "08_vif_check_v2.py",         # 11개 예측변수 간 다중공선성(VIF)을 확인하여 회귀계수 왜곡 여부 검정
    "09_additional_models.py"     # 로지스틱, 랜덤포레스트, XGBoost 3개 모델 선택 배경 및 추가 비교 분석 수행
]

for script in scripts:
    print(f"\n========== [실행 중] {script} ==========")
    
    # 해당 파이썬 스크립트 실행
    result = subprocess.run(["python", script])
    
    # 실행 중 에러가 발생하면 중단
    if result.returncode != 0:
        print(f" 에러 발생으로 중단됩니다: {script}")
        break

print("\n 모든 분석 과정이 완료되었습니다")
