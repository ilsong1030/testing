# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 07:18:51 2024

@author: User
"""
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Load the uploaded CSV file
file_path = 'C:/Users/User/Desktop/MM Marmosets/Marmoset exel/Restructured_Data_for_Analysis.csv'
data = pd.read_csv(file_path)

# Select only numerical columns for PCA
numerical_data = data.select_dtypes(include=['float64', 'int64'])

# Standardize the data (mean=0, variance=1)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

# Perform PCA
pca = PCA()
pca_result = pca.fit_transform(scaled_data)

# Create a DataFrame to display PCA results
pca_columns = [f'PC{i+1}' for i in range(pca_result.shape[1])]
pca_df = pd.DataFrame(pca_result, columns=pca_columns)

# Combine PCA results with original identifiers for context
pca_df_combined = pd.concat([data[['Observation id_', 'Conditions_', 'Subject_']], pca_df], axis=1)

# Explained Variance Summary
explained_variance = pd.DataFrame({
    'Principal Component': pca_columns,
    'Explained Variance Ratio': pca.explained_variance_ratio_,
    'Cumulative Variance': pca.explained_variance_ratio_.cumsum()
})

# Create Scree Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1))
plt.grid(True)
plt.show()

# Create Cumulative Variance Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Explained')
plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1))
plt.grid(True)
plt.show()

# Heatmap of variable contributions
loading_matrix = pd.DataFrame(pca.components_, columns=numerical_data.columns, index=[f'PC{i+1}' for i in range(pca.n_components_)])

# Rename columns in the loading matrix for clarity
renamed_columns = [
    col.replace('Total duration (s)_', 'Duration_').replace('Total number of occurences_', 'Frequency_') 
    for col in loading_matrix.columns
]
loading_matrix.columns = renamed_columns

# Replot the heatmap with updated column names
plt.figure(figsize=(12, 8))
sns.heatmap(loading_matrix, cmap='coolwarm', annot=False, cbar=True)
plt.title('Heatmap of Variable Contributions to Principal Components')
plt.xlabel('Original Variables (Renamed)')
plt.ylabel('Principal Components')
plt.show()

# 2D Scatter Plot of the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7, edgecolor='k')
plt.title('2D Scatter Plot of First Two Principal Components')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)
plt.show()

# 3D Scatter Plot of the first three principal components
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], alpha=0.7, edgecolor='k')
ax.set_title('3D Scatter Plot of First Three Principal Components')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()

# Create a heatmap for the original factors (numerical data)
renamed_columns_corr = [
    col.replace('Total duration (s)_', 'Duration_').replace('Total number of occurences_', 'Frequency_') 
    for col in numerical_data.columns
]
numerical_data_renamed = numerical_data.copy()
numerical_data_renamed.columns = renamed_columns_corr

plt.figure(figsize=(12, 8))
sns.heatmap(
    numerical_data_renamed.corr(),
    cmap='coolwarm',
    annot=False,
    cbar=True
)
plt.title('Heatmap of Correlations Between Original Factors')
plt.xlabel('Factors')
plt.ylabel('Factors')
plt.show()

# Create a Biplot (PC1 vs PC2 with loadings)
plt.figure(figsize=(12, 8))
# Scatter plot for observations
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7, label='')

# Add loadings as arrows
for i, (loading_x, loading_y) in enumerate(zip(loading_matrix.loc['PC1'], loading_matrix.loc['PC2'])):
    plt.arrow(0, 0, loading_x * 5, loading_y * 5, color='r', alpha=0.5)
    plt.text(loading_x * 5.2, loading_y * 5.2, loading_matrix.columns[i], color='r', fontsize=9)

plt.title('Biplot of PC1 and PC2 Components')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.axhline(0, color='gray', linewidth=0.8)
plt.axvline(0, color='gray', linewidth=0.8)
plt.grid(True)
plt.legend()
plt.show()

# Reattempt plotting the PC1-PC2 interaction values
plt.figure(figsize=(8, 6))

# Scatter plot for PC1 vs Interaction
plt.scatter(pca_df['PC1'], pca_df['PC1_PC2_Interaction'], alpha=0.7, label='PC1 vs Interaction', color='blue')

# Scatter plot for PC2 vs Interaction
plt.scatter(pca_df['PC2'], pca_df['PC1_PC2_Interaction'], alpha=0.7, label='PC2 vs Interaction', color='red')

# Add grid, labels, and legend
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.title('Interaction Value of PC1 and PC2')
plt.xlabel('PC1 and PC2 Values')
plt.ylabel('Interaction Value (PC1 * PC2)')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()