# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:50:28 2024

@author: User
"""
# Importing required libraries again as the session was reset
import pandas as pd

# Reload the file
file_path = '/Volumes/Extreme SSD/Master phase/MM Marmosets/Marmoset exel/Just Familiar vs Unfamiliar.csv'
data = pd.read_csv(file_path)

# Re-check the structure of the dataset
data.head()
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import os

# Function to calculate Mann-Whitney U test p-values
def mann_whitney_p(group, value_column):
    conditions = group['Conditions'].unique()
    if len(conditions) == 2:
        group1 = group[group['Conditions'] == conditions[0]][value_column]
        group2 = group[group['Conditions'] == conditions[1]][value_column]
        if group1.nunique() > 1 and group2.nunique() > 1:  # Ensure variance exists
            return mannwhitneyu(group1, group2, alternative='two-sided').pvalue
    return None

# Directory to save the plots
output_dir = '/Users/ilsongjeon/Desktop'
os.makedirs(output_dir, exist_ok=True)

# Prepare box plots for each behavior with improved handling for None p-values
for behavior in data['Behavior'].unique():
    subset = data[data['Behavior'] == behavior]

    # Calculate p-values for Duration and Frequency
    p_value_duration = mann_whitney_p(subset, 'Total duration (s)')
    p_value_frequency = mann_whitney_p(subset, 'Total number of occurences')

    plt.figure(figsize=(12, 5))

    # Boxplot for Duration
    plt.subplot(1, 2, 1)
    sns.boxplot(data=subset, x='Conditions', y='Total duration (s)', palette='Set2')
    duration_title = f"{behavior} Duration (p-value: {p_value_duration:.4f})" if p_value_duration is not None else f"{behavior} Duration (p-value: N/A)"
    plt.title(duration_title)
    plt.xlabel("Condition")
    plt.ylabel("Total Duration (s)")

    # Boxplot for Frequency
    plt.subplot(1, 2, 2)
    sns.boxplot(data=subset, x='Conditions', y='Total number of occurences', palette='Set2')
    frequency_title = f"{behavior} Frequency (p-value: {p_value_frequency:.4f})" if p_value_frequency is not None else f"{behavior} Frequency (p-value: N/A)"
    plt.title(frequency_title)
    plt.xlabel("Condition")
    plt.ylabel("Total Number of Occurrences")

    plt.tight_layout()
    plt.show()