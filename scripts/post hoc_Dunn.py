import pandas as pd
import scikit_posthocs as sp
import scipy.stats as stats

# 데이터 로드
file_path = '/Volumes/Extreme SSD/WienUni/Master Thesis (Marmosets)/MM Marmosets/Marmoset exel/Restructured Data for Analysis_zero_out.csv'
df = pd.read_csv(file_path)


# 분석할 변수 설정
conditions_col = "Conditions_"
behaviors = [
    "Total duration (s)_Gaze",
    "Total number of occurences_Head Swaying",
    "Total number of occurences_Tsik (Single Tsik)"
]

# Dunn's test 수행
dunn_results = {}

for behavior in behaviors:
    # 데이터 필터링
    data = df[[conditions_col, behavior]].dropna()
    
    # Dunn's test 수행 (Bonferroni 보정 적용)
    dunn_test = sp.posthoc_dunn(data, val_col=behavior, group_col=conditions_col, p_adjust='bonferroni')
    
    # 결과 저장
    dunn_results[behavior] = dunn_test
    
    # CSV 파일로 저장
    file_name = f"Dunn_test_{behavior.replace(' ', '_').replace('/', '_')}.csv"
    dunn_test.to_csv(file_name)

# 결과 출력
for behavior, result in dunn_results.items():
    print(f"\nDunn’s test results for {behavior}:")
    print(result)
    
