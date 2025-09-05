#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 20:15:43 2025

@author: ilsongjeon
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 파일 경로 설정
file_paths = {
    "Gaze Duration": '/Volumes/Extreme SSD/WienUni/Master Thesis (Marmosets)/MM Marmosets/Plots_and_coding/Dunn_test_Total_duration_(s)_Gaze.csv',
    "Head Swaying Frequency": '/Volumes/Extreme SSD/WienUni/Master Thesis (Marmosets)/MM Marmosets/Plots_and_coding/Dunn_test_Total_number_of_occurences_Head_Swaying.csv',
    "Tsik(Single Tsik) Frequency": '/Volumes/Extreme SSD/WienUni/Master Thesis (Marmosets)/MM Marmosets/Plots_and_coding/Dunn_test_Total_number_of_occurences_Tsik_(Single_Tsik).csv'
}

# Dunn’s test 결과 로드
dunn_results = {key: pd.read_csv(path, index_col=0) for key, path in file_paths.items()}

# 원본 데이터 로드
df = pd.read_csv('/Volumes/Extreme SSD/WienUni/Master Thesis (Marmosets)/MM Marmosets/Marmoset exel/Restructured Data for Analysis_zero_out.csv')

# 분석할 변수 매칭
behavior_mapping = {
    "Gaze Duration": "Total duration (s)_Gaze",
    "Head Swaying Frequency": "Total number of occurences_Head Swaying",
    "Tsik(Single Tsik) Frequency": "Total number of occurences_Tsik (Single Tsik)"
}

# 조건 추출
conditions = df["Conditions_"].unique()

# Dunn’s test에서 유의미한 비교 찾기 (p-value < 0.05)
def find_significant_pairs(dunn_df):
    significant_pairs = []
    for i, cond1 in enumerate(conditions):
        for j, cond2 in enumerate(conditions):
            if i < j:  # 중복 피하기
                p_value = dunn_df.loc[cond1, cond2]
                if p_value < 0.05:
                    significant_pairs.append((cond1, cond2, p_value))
    return significant_pairs

# 조건별 색상 매핑
condition_column = "Conditions_"
conditions = df[condition_column].unique()
colors = {cond: color for cond, color in zip(conditions, ["red", "blue", "yellow", "green"])}


# Boxplot + Dunn’s test 결과 시각화 (수정된 버전)
for behavior, behavior_col in behavior_mapping.items():
    plt.figure(figsize=(10, 6))
    
    # Boxplot 생성 (조건별 색상 적용)
    ax = sns.boxplot(data=df, x=condition_column, y=behavior_col, palette=colors)
    
    # Dunn’s test 결과 가져오기
    significant_pairs = find_significant_pairs(dunn_results[behavior])

    # 유의미한 차이를 선과 별표로 표시
    y_max = df[behavior_col].max()  # Y축 최대값
    y_offset = y_max * 0.1  # 선과 별표의 높이 조정

    for i, (cond1, cond2, p_value) in enumerate(significant_pairs):
        x1, x2 = np.where(conditions == cond1)[0][0], np.where(conditions == cond2)[0][0]
        y = y_max + (i+1) * y_offset  # 선 위치 조정
        
        # 선 그리기
        plt.plot([x1, x1, x2, x2], [y, y + 0.02, y + 0.02, y], color="black")
        
        # 유의 수준에 따른 별표 표시
        if p_value < 0.001:
            plt.text((x1 + x2) / 2, y + 0.02, "***", ha="center", fontsize=12, color="black")
        elif p_value < 0.01:
            plt.text((x1 + x2) / 2, y + 0.02, "**", ha="center", fontsize=12, color="black")
        elif p_value < 0.05:
            plt.text((x1 + x2) / 2, y + 0.02, "*", ha="center", fontsize=12, color="black")

    # 그래프 제목 및 라벨 설정
    plt.title(f"{behavior} (Dunn's Test Significant Differences)")
    plt.xlabel("Conditions_")
    
    # Y축 라벨을 'Duration' 또는 'Frequency'로 설정
    if "duration" in behavior.lower():
        plt.ylabel("Duration")
    else:
        plt.ylabel("Frequency")
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()