# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:50:28 2024

@author: User
"""
# 필요한 라이브러리 임포트
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import wilcoxon

# 원본 데이터 로드
file_path = '/Volumes/Extreme SSD/WienUni/Master Thesis (Marmosets)/MM Marmosets/Marmoset exel/Restructured Data for Analysis_zero_out.csv'
df = pd.read_csv(file_path)

# 📌 분석할 행동 변수 (예: Gaze Duration, Head Swaying Frequency, Tsik Frequency)
behaviors = [
    "Total duration (s)_Gaze",
    "Total number of occurences_Head Swaying",
    "Total number of occurences_Tsik (Single Tsik)"
]

# 📌 Familiar 조건과 Unfamiliar 조건 정의
condition_familiar = ["Familiar_match", "Familiar_unmatch"]
condition_unfamiliar = ["Unfamiliar_match", "Unfamiliar_unmatch"]

# 📌 데이터 타입 변환 (문자열 → 숫자)
for behavior in behaviors:
    df[behavior] = pd.to_numeric(df[behavior], errors="coerce")

# 📌 개체별 Familiar 조건과 Unfamiliar 조건 평균 계산
df_grouped = df.groupby(["Subject_", "Conditions_"]).mean(numeric_only=True).reset_index()

# 📌 Wilcoxon Signed-Rank Test 결과 저장
wilcoxon_results = {}

for behavior in behaviors:
    # Familiar 그룹 평균
    familiar_data = df_grouped[df_grouped["Conditions_"].isin(condition_familiar)].groupby("Subject_")[behavior].mean()
    
    # Unfamiliar 그룹 평균
    unfamiliar_data = df_grouped[df_grouped["Conditions_"].isin(condition_unfamiliar)].groupby("Subject_")[behavior].mean()
    
    # NaN 값 제거 (필수)
    familiar_data = familiar_data.dropna()
    unfamiliar_data = unfamiliar_data.dropna()
    
    # Wilcoxon Signed-Rank Test 수행 (데이터 크기가 0이 아닌 경우만 실행)
    if len(familiar_data) == len(unfamiliar_data) and len(familiar_data) > 0:
        stat, p_value = wilcoxon(familiar_data, unfamiliar_data)
        wilcoxon_results[behavior] = round(p_value, 4)
    else:
        wilcoxon_results[behavior] = None  # 데이터 부족 시 None

# 📌 시각화 코드 (Wilcoxon p-value 추가)
for behavior in behaviors:
    plt.figure(figsize=(8, 6))
    
    # Familiar vs Unfamiliar 그룹 분류
    df_plot = df.copy()
    df_plot["Familiarity"] = df_plot["Conditions_"].apply(lambda x: "Familiar" if x in condition_familiar else "Unfamiliar")
    
    # 숫자 변환 및 NaN 제거
    df_plot[behavior] = pd.to_numeric(df_plot[behavior], errors="coerce")
    df_plot = df_plot.dropna(subset=[behavior])
    
    # Boxplot 생성 (조건별 색상 적용)
    sns.boxplot(data=df_plot, x="Familiarity", y=behavior, palette={"Familiar": "blue", "Unfamiliar": "red"})
    
    # Y축 라벨 수정 (Duration vs Frequency)
    y_label = "Duration" if "duration" in behavior.lower() else "Frequency"
    plt.ylabel(y_label)
    
     # 그래프 제목 설정 (Wilcoxon p-value 포함, 형식 수정)
    behavior_label = "Gaze Duration" if "Gaze" in behavior else \
                     "Head Swaying Frequency" if "Head Swaying" in behavior else \
                     "Tsik (Single Tsik) Frequency"
    
    p_value_text = f"Wilcoxon p = {wilcoxon_results[behavior]:.4f}" if wilcoxon_results[behavior] is not None else "Wilcoxon p = N/A"
    plt.title(f"{behavior_label} by Familiarity ({p_value_text})")

    plt.xlabel("Familiarity")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
