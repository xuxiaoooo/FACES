import pandas as pd
from datetime import datetime

# Load the CSV data
file_path = '../data/baseinfo.csv'
data = pd.read_csv(file_path)

# Define a function to calculate age based on a given 'survey_date'
def calculate_age(birthdate, survey_date):
    birthdate = datetime.strptime(str(birthdate), "%Y%m%d")
    survey_date = datetime.strptime(str(survey_date), "%Y%m%d")
    age = survey_date.year - birthdate.year - ((survey_date.month, survey_date.day) < (birthdate.month, birthdate.day))
    return max(10, min(age, 18))  # Age is capped at 10 and 18

# Add 'age' column to the dataframe
survey_date = 20221101
data['age'] = data['birthday'].apply(lambda x: calculate_age(x, survey_date))

# Calculate the gender distribution
gender_counts = data['gender'].value_counts().to_frame()
gender_counts.columns = ['No. of samples (Mean)']
gender_counts['Percentage (Standard Deviation)'] = gender_counts['No. of samples (Mean)'] / gender_counts['No. of samples (Mean)'].sum() * 100

# Calculate the age distribution
age_counts = data['age'].value_counts().sort_index().to_frame()
age_counts.columns = ['No. of samples (Mean)']
age_counts['Percentage (Standard Deviation)'] = age_counts['No. of samples (Mean)'] / age_counts['No. of samples (Mean)'].sum() * 100

# Calculate the grade distribution
grade_mapping = {
    '小学一年级': 1, '小学二年级': 2, '小学三年级': 3, '小学四年级': 4, '小学五年级': 5, '小学六年级': 6,
    '初中一年级': 7, '初中二年级': 8, '初中三年级': 9,
    '高中一年级': 10, '高中二年级': 11, '高中三年级': 12
}

# Map the 'grade_name' to numeric grades
data['grade'] = data['grade_name'].map(grade_mapping)

# Calculate the grade distribution
grade_counts = data['grade'].value_counts().sort_index().to_frame()
grade_counts.columns = ['No. of samples (Mean)']
grade_counts['Percentage (Standard Deviation)'] = grade_counts['No. of samples (Mean)'] / grade_counts['No. of samples (Mean)'].sum() * 100

# Combine all the distributions into a single dataframe for display
combined_counts = pd.concat([gender_counts, age_counts, grade_counts], keys=['Gender', 'Age', 'Grade'])

combined_counts.reset_index(inplace=True)
combined_counts.columns = ['Characteristics', 'Detail', 'No. of samples (Mean)', 'Percentage (Standard Deviation)']

# Calculate the mean age and its standard deviation
mean_age = data['age'].mean()
std_age = data['age'].std()

# Insert mean age row
mean_age_row = pd.DataFrame([['Age', 'Mean age, years', mean_age, std_age]], columns=combined_counts.columns)
combined_counts = pd.concat([mean_age_row, combined_counts], ignore_index=True)

# Reorder the dataframe to have 'Gender', 'Age', and then 'Grade'
reordered_df = pd.concat([combined_counts[combined_counts['Characteristics'] == 'Gender'],
                          combined_counts[combined_counts['Characteristics'] == 'Age'][1:], # Exclude the mean age row
                          combined_counts[combined_counts['Characteristics'] == 'Grade']])

# Adding the mean age row at the correct position
reordered_df = pd.concat([reordered_df.iloc[:1], mean_age_row, reordered_df.iloc[1:]]).reset_index(drop=True)

print(reordered_df)