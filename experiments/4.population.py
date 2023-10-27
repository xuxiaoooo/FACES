import pandas as pd
from datetime import datetime

class PopulationAnalysis:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.today = datetime.strptime("20221101", "%Y%m%d")

    def gender_distribution(self):
        return self.df['gender'].value_counts()

    def grade_distribution(self):
        return self.df['grade_name'].value_counts()

    def class_distribution(self):
        return self.df['class_name'].value_counts()

    def org_distribution(self):
        return self.df['org_name'].value_counts()

    def age_distribution(self):
        problematic_entries = []

        def compute_age(x):
            try:
                age = (self.today - datetime.strptime(str(x), "%Y%m%d")).days // 365
                return age
            except ValueError:
                problematic_entries.append(x)
                return None  # 返回 None，以便稍后我们可以更容易地识别出问题的条目

        self.df['age'] = self.df['birthday'].apply(compute_age)

        if problematic_entries:
            print(f"Problematic birthday entries: {problematic_entries}")

        return self.df['age'].value_counts().sort_index()

    def ethnic_distribution(self):
        return self.df['ethnic'].value_counts()

    def dass_scores(self):
        depression_categories = ["Normal", "Mild", "Moderate", "Severe", "Extremely Severe"]
        anxiety_categories = ["Normal", "Mild", "Moderate", "Severe", "Extremely Severe"]
        stress_categories = ["Normal", "Mild", "Moderate", "Severe", "Extremely Severe"]

        self.df['depression_label'] = pd.cut(self.df['depression'], bins=[-1, 9, 13, 20, 27, float('inf')], labels=depression_categories)
        self.df['anxiety_label'] = pd.cut(self.df['anxiety'], bins=[-1, 7, 9, 14, 19, float('inf')], labels=anxiety_categories)
        self.df['stress_label'] = pd.cut(self.df['stress'], bins=[-1, 14, 18, 25, 33, float('inf')], labels=stress_categories)

        return {
            "Depression": self.df['depression_label'].value_counts(),
            "Anxiety": self.df['anxiety_label'].value_counts(),
            "Stress": self.df['stress_label'].value_counts()
        }

    def analyze(self):
        results = {
            "Gender Distribution": self.gender_distribution(),
            "Grade Distribution": self.grade_distribution(),
            "Class Distribution": self.class_distribution(),
            "Organization Distribution": self.org_distribution(),
            "Age Distribution": self.age_distribution(),
            "Ethnic Distribution": self.ethnic_distribution(),
            "DASS Scores": self.dass_scores()
        }
        return results

if __name__ == "__main__":
    analyzer = PopulationAnalysis('../data/baseinfo.csv')
    results = analyzer.analyze()
    for key, value in results.items():
        print(f"\n{key}:\n")
        print(value)