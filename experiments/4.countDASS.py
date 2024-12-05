import pandas as pd

file_path = '../data/baseinfo.csv'

# Read the CSV file
data = pd.read_csv(file_path)

# Define the boundaries for the categories
depression_boundaries = [9, 13, 20, 27]
anxiety_boundaries = [7, 9, 14, 19]
stress_boundaries = [14, 18, 25, 33]

# Function to categorize the scores
def categorize_scores(scores, boundaries):
    categories = ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe']
    score_distribution = {category: 0 for category in categories}

    for score in scores:
        for i, boundary in enumerate(boundaries):
            if score <= boundary:
                score_distribution[categories[i]] += 1
                break
        else:
            score_distribution['Extremely Severe'] += 1
            
    return score_distribution

# Calculate the distribution for Depression, Anxiety, and Stress
depression_distribution = categorize_scores(data['Depression'], depression_boundaries)
anxiety_distribution = categorize_scores(data['Anxiety'], anxiety_boundaries)
stress_distribution = categorize_scores(data['Stress'], stress_boundaries)

# Print the distributions
print("Depression Score Distribution:", depression_distribution)
print("Anxiety Score Distribution:", anxiety_distribution)
print("Stress Score Distribution:", stress_distribution)
