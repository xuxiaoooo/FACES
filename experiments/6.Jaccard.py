import pandas as pd
from sklearn.metrics import jaccard_score
import numpy as np

# Load the dataset
file_path = '../data/label.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Function to calculate Jaccard scores for the first task
def calculate_jaccard_scores_task1(data, columns):
    class_mapping = {0: 0, 1: 1, 2: 1}
    scores = pd.DataFrame(index=columns, columns=columns)

    for col1 in columns:
        for col2 in columns:
            if col1 == col2:
                scores.loc[col1, col2] = 1.0
            else:
                # Apply class mapping to create binary classes
                data_col1 = data[col1].map(class_mapping)
                data_col2 = data[col2].map(class_mapping)
                # Calculate Jaccard score
                score = jaccard_score(data_col1, data_col2)
                scores.loc[col1, col2] = score

    return scores

# Function to calculate Jaccard scores for the second task
def calculate_jaccard_scores_task2(data, columns):
    class_mapping = {0: 0, 2: 1}

    # Exclude rows with value 1 in any of the columns
    data_filtered = data[~data[columns].isin([1]).any(axis=1)]
    print(len(data_filtered))
    scores = pd.DataFrame(index=columns, columns=columns)
    for col1 in columns:
        for col2 in columns:
            if col1 == col2:
                scores.loc[col1, col2] = 1.0
            else:
                # Apply class mapping to create binary classes
                data_col1 = data_filtered[col1].map(class_mapping)
                data_col2 = data_filtered[col2].map(class_mapping)
                # Calculate Jaccard score
                score = jaccard_score(data_col1, data_col2)
                scores.loc[col1, col2] = score

    return scores

# Define the columns for each task
columns_task1 = ['Depression', 'Anxiety', 'Stress', 'Cluster_2']
columns_task2 = ['Depression', 'Anxiety', 'Stress', 'Cluster_3']

# Calculate Jaccard scores for each task
jaccard_scores_task1 = calculate_jaccard_scores_task1(data, columns_task1)
jaccard_scores_task2 = calculate_jaccard_scores_task2(data, columns_task2)

# Round the scores to three decimal places
jaccard_scores_task1 = jaccard_scores_task1.astype(float).round(3)
jaccard_scores_task2 = jaccard_scores_task2.astype(float).round(3)

# Print the Jaccard scores
print("Jaccard scores for Task 1:")
print(jaccard_scores_task1)
print("\nJaccard scores for Task 2:")
print(jaccard_scores_task2)
