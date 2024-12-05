import pandas as pd

class DASS21:
    def __init__(self, df_base, df_cl):
        self.df = pd.merge(df_base, df_cl, on='cust_id')
    
    @staticmethod
    def classify_score(score, category):
        # Depression classification
        if category == 'Depression':
            if 0 <= score <= 9:
                return 0
            elif 10 <= score <= 13:
                return 1
            else:
                return 2
        # Anxiety classification
        elif category == 'Anxiety':
            if 0 <= score <= 7:
                return 0
            elif 8 <= score <= 9:
                return 1
            else:
                return 2
        # Stress classification
        elif category == 'Stress':
            if 0 <= score <= 14:
                return 0
            elif 15 <= score <= 18:
                return 1
            else:
                return 2
    
    def classify(self):
        self.df['Depression'] = self.df['Depression'].apply(lambda x: self.classify_score(x, 'Depression'))
        self.df['Anxiety'] = self.df['Anxiety'].apply(lambda x: self.classify_score(x, 'Anxiety'))
        self.df['Stress'] = self.df['Stress'].apply(lambda x: self.classify_score(x, 'Stress'))
        return self.df

if __name__ == '__main__':
    lb_base = pd.read_csv('../data/baseinfo.csv')[['cust_id', 'Depression', 'Anxiety', 'Stress']]
    lb_cl = pd.read_csv('../data/cluster_label.csv')
    classifier = DASS21(lb_base, lb_cl)
    result = classifier.classify()
    result.to_csv('../data/label.csv', index=False)
